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
YOLOE_IMAGE="z-mobile-manip-yoloe:latest"
USER_SYSTEMD_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UI_PORT="${Z_MANIP_DEBUG_UI_PORT:-8766}"
NUC_HOST="${GO2W_NUC_HOST:-yusenzlabnuc@192.168.3.8}"
NUC_KEY="${GO2W_NUC_SSH_KEY:-$HOME/.ssh/id_ed25519_codex_nuc}"
RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp}"
LOCK_FILE="$RUNTIME_DIR/z-manip-component-manager.lock"
LOG_ROOT="${Z_MANIP_COMPONENT_LOG_ROOT:-$WORKSPACE_ROOT/artifacts/go2w_real/component_logs}"
MANAGER_LOG="$LOG_ROOT/component-manager.log"
CAMERA_ARTIFACT="${Z_MANIP_CAMERA_ARTIFACT:-$WORKSPACE_ROOT/artifacts/go2w_real/latest/camera-latest.jpg}"
PERCEPTION_RUNNER_SOCKET="$WORKSPACE_ROOT/artifacts/go2w_real/.perception_runner.sock"
PLANNING_RUNNER_SOCKET="$WORKSPACE_ROOT/artifacts/go2w_real/.planning_runner_scratch/.planner.sock"
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
       go2w_component_manager.sh shutdown
       go2w_component_manager.sh install
       go2w_component_manager.sh status [all|ui|nuc-camera|passive-feedback|observer|rgbd|grounding|edgetam|perception|perception-all|reactive-control|posture-bridge|whole-body|mobile-control]
       go2w_component_manager.sh restart {ui|nuc-camera|passive-feedback|observer|rgbd|grounding|edgetam|perception|perception-all|reactive-control|posture-bridge|whole-body|mobile-control}
       go2w_component_manager.sh logs {manager|ui|nuc-camera|passive-feedback|observer|rgbd|grounding|edgetam|perception|perception-all|reactive-control|posture-bridge|mobile-control} [lines]
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
    ui|nuc-camera|passive-feedback|observer|rgbd|grounding|edgetam|perception|perception-all|reactive-control|posture-bridge|whole-body|mobile-control) return 0 ;;
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

runtime_fingerprint() {
  "$LAB_SCRIPT" fingerprint
}

container_runtime_fingerprint() {
  $DOCKER inspect "$1" \
    --format '{{ index .Config.Labels "org.zlab.z-manip.runtime-sha256" }}' \
    2>/dev/null || true
}

runner_socket_private() {
  local path="$1" mode owner group
  [[ -S "$path" && ! -L "$path" ]] || return 1
  owner="$(stat -c %u "$path" 2>/dev/null)" || return 1
  group="$(stat -c %g "$path" 2>/dev/null)" || return 1
  mode="$(stat -c %a "$path" 2>/dev/null)" || return 1
  [[ "$owner" == "$(id -u)" && "$group" == "$(id -g)" ]] || return 1
  (( (8#$mode & 077) == 0 ))
}

resident_runners_current() {
  local expected perception planning
  expected="$(runtime_fingerprint)" || return 1
  perception="$(container_runtime_fingerprint z-manip-perception-runner)"
  planning="$(container_runtime_fingerprint z-manip-planning-runner)"
  [[ -n "$expected" && "$perception" == "$expected" && "$planning" == "$expected" ]] \
    && runner_socket_private "$PERCEPTION_RUNNER_SOCKET" \
    && runner_socket_private "$PLANNING_RUNNER_SOCKET"
}

yoloe_source_hash() {
  local files=(
    "$STACK_ROOT/docker/yoloe_service/Dockerfile"
    "$STACK_ROOT/docker/yoloe_service/requirements.txt"
    "$STACK_ROOT/scripts/runtime/local_grounding_service.py"
  )
  local file
  for file in "${files[@]}"; do
    [[ -f "$file" ]] || return 1
  done
  sha256sum "${files[@]}" | sha256sum | cut -d' ' -f1
}

yoloe_image_current() {
  local expected actual
  expected="$(yoloe_source_hash)" || return 1
  actual="$($DOCKER image inspect "$YOLOE_IMAGE" \
    --format '{{ index .Config.Labels "org.zlab.yoloe.source-sha256" }}' 2>/dev/null || true)"
  [[ -n "$expected" && "$actual" == "$expected" ]]
}

build_yoloe_image() {
  local source_hash
  source_hash="$(yoloe_source_hash)" || return 1
  $DOCKER build -f "$STACK_ROOT/docker/yoloe_service/Dockerfile" \
    --label "org.zlab.yoloe.source-sha256=$source_hash" \
    -t "$YOLOE_IMAGE" "$STACK_ROOT"
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
    grounding)
      if $SYSTEMCTL --user is-active --quiet "$GROUNDING_UNIT" 2>/dev/null; then
        if grounding_ready; then
          state="healthy"
          summary="YOLOE-11S CUDA HTTP health OK"
        else
          state="degraded"
          summary="YOLOE service active but model health unavailable"
        fi
      else
        summary="YOLOE grounding user service inactive"
      fi
      ;;
    perception)
      if container_running z-manip-hw \
          && container_running z-manip-perception-runner \
          && container_running z-manip-planning-runner; then
        local expected perception_hash planning_hash
        expected="$(runtime_fingerprint 2>/dev/null || true)"
        perception_hash="$(container_runtime_fingerprint z-manip-perception-runner)"
        planning_hash="$(container_runtime_fingerprint z-manip-planning-runner)"
        if resident_runners_current; then
          state="healthy"
          summary="ROS $(container_summary z-manip-hw); warm runners current fingerprint=${expected:0:12}"
        else
          state="degraded"
          summary="warm runner stale/unsafe: expected=${expected:0:12}, perception=${perception_hash:0:12}, planning=${planning_hash:0:12}; run 'manip component restart perception'"
        fi
      else
        summary="ROS $(container_summary z-manip-hw); perception runner $(container_summary z-manip-perception-runner); planning runner $(container_summary z-manip-planning-runner)"
      fi
      ;;
    perception-all)
      local observer_state rgbd_state grounding_state edgetam_state perception_state
      observer_state="$(status_state observer)"
      rgbd_state="$(status_state rgbd)"
      grounding_state="$(status_state grounding)"
      edgetam_state="$(status_state edgetam)"
      perception_state="$(status_state perception)"
      if [[ "$observer_state" == healthy && "$rgbd_state" == healthy && "$grounding_state" == healthy && "$edgetam_state" == healthy && "$perception_state" == healthy ]]; then
        state="healthy"
        summary="observer + RGB-D + YOLOE + EdgeTAM + perception ROS healthy"
      elif [[ "$observer_state" == offline || "$rgbd_state" == offline || "$grounding_state" == offline || "$edgetam_state" == offline || "$perception_state" == offline ]]; then
        state="offline"
        summary="observer=$observer_state, rgbd=$rgbd_state, grounding=$grounding_state, edgetam=$edgetam_state, perception=$perception_state"
      else
        state="degraded"
        summary="observer=$observer_state, rgbd=$rgbd_state, grounding=$grounding_state, edgetam=$edgetam_state, perception=$perception_state"
      fi
      ;;
    reactive-control)
      if remote_service_active "$REACTIVE_NUC_UNIT"; then
        state="healthy"
        summary="NUC single WebRTC owner active; Move ready, Euler capability negotiated at runtime"
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
        summary="NUC owner + capability-aware posture relay + Pinocchio/CasADi ready"
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

grounding_ready() {
  local health
  health="$($CURL -fsS --max-time 3 http://127.0.0.1:8771/health 2>/dev/null || true)"
  [[ "$health" == *'"ready": true'* || "$health" == *'"ready":true'* ]] \
    && [[ "$health" == *'"backend": "yoloe"'* || "$health" == *'"backend":"yoloe"'* ]]
}

perception_ready() {
  if ! container_running z-manip-hw; then
    return 1
  fi
  if ! container_running z-manip-perception-runner; then
    return 1
  fi
  if ! container_running z-manip-planning-runner; then
    return 1
  fi
  if ! resident_runners_current; then
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
    grounding)
      if ! yoloe_image_current; then
        build_yoloe_image || return 1
      fi
      install_pc_units || return 1
      $SYSTEMCTL --user restart "$GROUNDING_UNIT" || return 1
      wait_until "YOLOE grounding" grounding_ready
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
      restart_one grounding || return 1
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

bringup_rgbd_with_retry() {
  # Truly-cold RGB-D bringup races DDS discovery / USB enumeration against the
  # camera-artifact freshness gate, so the first restart+wait can time out even
  # though the camera is fine (component-manager.log 2026-07-23: two morning
  # episodes needed 2-3 operator re-runs of cold bringup). Absorb that with one
  # bounded container restart + re-wait before declaring failure; a genuinely
  # dead camera still fails closed after this single retry.
  if restart_one rgbd; then
    return 0
  fi
  # The retry only helps a discovery/enumeration race; a physically absent
  # camera must keep its fast, legible failure instead of hiding behind a
  # container restart.
  if ! remote_camera_device_ready; then
    printf '[%s] cold bringup: D435 USB device is absent; skipping the RGB-D restart retry\n' \
      "$(date --iso-8601=seconds)" >&2
    return 1
  fi
  printf '[%s] cold bringup self-heal: RGB-D not ready on first pass; trigger=rgbd_ready_timeout action=restart z-manip-rgbd container once and re-wait\n' \
    "$(date --iso-8601=seconds)" >&2
  "$LAB_SCRIPT" restart-rgbd || return 1
  wait_until "RGB-D bridge" rgbd_ready
}

shutdown_stack() {
  printf '[%s] full-stack shutdown begin\n' "$(date --iso-8601=seconds)"
  # The UI owns workflows: stop it first so nothing new can start mid-way.
  # Every stop below is a clean systemd/docker stop -- a process hard-killed
  # while holding a lock poisons shared state (JACK /dev/shm, 2026-07-23).
  $SYSTEMCTL --user stop "$UI_UNIT" 2>/dev/null || true
  $SYSTEMCTL --user stop "$OBSERVER_UNIT" "$GROUNDING_UNIT" "$POSTURE_UNIT" 2>/dev/null || true
  "$LAB_SCRIPT" stop >/dev/null 2>&1 || true
  $DOCKER stop z-manip-edgetam z-mobile-manip-posture-intent z-manip-runtime-observer >/dev/null 2>&1 || true
  if remote_command 'systemctl --user stop z-mobile-manip-go2w-reactive-live.service z-mobile-manip-piper-reactive-view.service z-manip-piper-passive-feedback.service d435i.service webrtc-video.service 2>/dev/null; true' >/dev/null 2>&1; then
    printf 'NUC robot services stopped cleanly\n'
  else
    printf 'NUC unreachable; its services were not stopped\n' >&2
  fi
  printf '[%s] full-stack shutdown complete\n' "$(date --iso-8601=seconds)"
}

cold_bringup_steps() {
  printf '[%s] cold bringup begin\n' "$(date --iso-8601=seconds)"
  install_pc_units || return 1
  # A process killed while initialising the JACK client registry leaves a
  # poisoned BDB mutex in /dev/shm and every later WebRTC-stack import on the
  # NUC hangs before its first log line (2026-07-23). Clear the stale
  # registry only while the WebRTC owner is down.
  remote_command 'systemctl --user is-active --quiet z-mobile-manip-go2w-reactive-live.service || rm -rf /dev/shm/jack_db-1000' >/dev/null 2>&1 || true
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
  restart_one grounding || return 1
  restart_one edgetam || return 1
  # The RGB-D health gate reads camera-latest.jpg, which the runtime observer
  # writes.  On a genuinely cold start (manip stop also stops the observer)
  # the observer must be running before the RGB-D wait can ever pass.
  $SYSTEMCTL --user enable --now "$OBSERVER_UNIT" || return 1
  wait_until "runtime observer" observer_ready || return 1
  bringup_rgbd_with_retry || return 1
  restart_one perception || return 1
  # ``enable --now`` leaves an already-running Python UI on its old imported
  # modules.  Cold bringup is also the supported post-update reload path, so
  # force one bounded UI restart after the resident workers are recreated.
  $SYSTEMCTL --user enable "$UI_UNIT" || return 1
  $SYSTEMCTL --user restart "$UI_UNIT" || return 1
  wait_until "UI workbench" ui_ready || return 1
  "$REACTIVE_INSTALLER" || return 1
  wait_until "NUC reactive control" remote_service_active "$REACTIVE_NUC_UNIT" || return 1
  wait_until "posture relay" posture_bridge_ready || return 1
  if ! $DOCKER image inspect "$WHOLE_BODY_IMAGE" >/dev/null 2>&1; then
    $DOCKER build -t "$WHOLE_BODY_IMAGE" "$STACK_ROOT/docker/whole_body_runtime" || return 1
  fi
  if $DOCKER exec z-manip-hw bash -lc \
      'source /opt/ros/jazzy/setup.bash && timeout 10 ros2 topic echo /go2w/posture_state std_msgs/msg/String --once' \
      >/dev/null 2>&1; then
    printf 'NUC WebRTC owner is publishing posture state\n'
  else
    printf 'WARNING: /go2w/posture_state is silent after bringup; check the NUC reactive-live journal (stale JACK lock? robot powered off?)\n' >&2
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
    grounding) $JOURNALCTL --user -u "$GROUNDING_UNIT" -n "$lines" --no-pager -o cat 2>&1 ;;
    perception) $DOCKER logs --tail "$lines" z-manip-hw 2>&1 ;;
    reactive-control) remote_command "journalctl --user -u $REACTIVE_NUC_UNIT -n $lines --no-pager -o cat" 2>&1 ;;
    posture-bridge) $JOURNALCTL --user -u "$POSTURE_UNIT" -n "$lines" --no-pager -o cat 2>&1 ;;
    perception-all)
      for item in observer grounding edgetam rgbd perception; do
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

# Allow tests to source this script for function-level verification without
# triggering the CLI dispatch below. When sourced, `return` succeeds at the top
# level; when executed, it fails and the dispatch runs normally.
(return 0 2>/dev/null) && __sourced=1 || __sourced=0
if ((__sourced)); then
  return 0
fi

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
      for item in ui nuc-camera passive-feedback observer rgbd grounding edgetam perception perception-all reactive-control posture-bridge whole-body mobile-control; do
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
  shutdown)
    [[ "$component" == all ]] || usage
    exec 9>"$LOCK_FILE"
    if ! flock -n 9; then
      printf 'another component restart is already running\n' >&2
      exit 75
    fi
    shutdown_stack 2>&1 | tee -a "$MANAGER_LOG"
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
    for item in ui nuc-camera passive-feedback observer rgbd grounding edgetam perception perception-all reactive-control posture-bridge whole-body mobile-control; do
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
