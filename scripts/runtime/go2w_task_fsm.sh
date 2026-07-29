#!/usr/bin/env bash
set -euo pipefail

# Fixed-surface lifecycle manager for the mobile-manipulation task FSM.
#
# It owns exactly ONE dedicated container ("z-manip-task") that runs the
# singleton supervisor (scripts/runtime/mobile_manipulation_supervisor.py,
# installed in the runtime image as /usr/local/bin/z-manip-mobile-manipulation).
# The supervisor in turn owns one `ros2 launch z_manip_task
# mobile_manipulation.launch.py`, which brings up the perception, coarse-
# navigation bridge, MoveIt, observed-place, and task-runtime nodes as a
# fail-closed singleton (a deployment-scoped flock + an empty-graph preflight;
# it REFUSES to start when a critical producer -- e.g. a lab perception
# `z_manip_edgetam` -- is already on the domain, and never terminates a
# producer it did not launch).
#
# ZERO base-chain by construction: this manager holds no actuator transport,
# never starts/stops the NUC reactive/base-control services, and accepts only
# the fixed verbs below. The task FSM's own velocity output goes to the
# platform `local_velocity` topic, which has NO consumer until a base chain
# (`manip start_base` / `manip component restart reactive-control`) is
# deliberately enabled. Bringing the FSM up is therefore motion-inert on its
# own; the chassis moves only once the separate base chain is enabled.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STACK_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

# --- Injectable surface (tests override these; live uses the defaults) --------
DOCKER="${Z_MANIP_DOCKER_BIN:-docker}"
CONTAINER="${Z_MANIP_TASK_CONTAINER:-z-manip-task}"
IMAGE="${Z_MANIP_RUNTIME_IMAGE:-z-manip-runtime:pinocchio}"
DOMAIN_ID="${ROS_DOMAIN_ID:-20}"
NAMESPACE="${Z_MANIP_ROS_NAMESPACE:-/}"
DDS_CONFIG="${Z_MANIP_DDS_CONFIG:-$STACK_ROOT/docker/runtime/cyclonedds-go2w-pc.xml}"
ROBOT_ASSETS="${Z_MANIP_ROBOT_ASSETS:-$STACK_ROOT/../go2W_Sim/assets}"

# In-container deployment paths (baked into the runtime image / mounted below).
IN_STACK_CONFIG="${Z_MANIP_STACK_CONFIG:-/opt/z_manip/configs/go2w_piper.json}"
IN_ROBOT_URDF="${Z_MANIP_ROBOT_URDF:-/robot/assets/urdf/go2w_sensored.urdf}"
IN_COLLISION_MODEL="${Z_MANIP_COLLISION_MODEL_FILE:-/opt/z_manip/configs/piper_collision_capsules.json}"
SUPERVISOR_BIN="${Z_MANIP_SUPERVISOR_BIN:-/usr/local/bin/z-manip-mobile-manipulation}"

STARTUP_TIMEOUT="${Z_MANIP_TASK_STARTUP_TIMEOUT:-90}"
SHUTDOWN_TIMEOUT="${Z_MANIP_TASK_SHUTDOWN_TIMEOUT:-12}"
STATUS_TOPIC="${Z_MANIP_TASK_STATUS_TOPIC:-/z_manip/task/status}"
READY_MARKER='[z-manip-supervisor] uniquely ready'

# Private environment file (HTTP service URLs, etc.), matching the perception
# lab's resolution order. Optional -- the launch has generic defaults.
if [[ -n "${Z_MANIP_ENV_FILE:-}" ]]; then
  ENV_FILE="$Z_MANIP_ENV_FILE"
elif [[ -f "$STACK_ROOT/.env" ]]; then
  ENV_FILE="$STACK_ROOT/.env"
elif [[ -f "$STACK_ROOT/../z-agent/.env" ]]; then
  ENV_FILE="$STACK_ROOT/../z-agent/.env"
else
  ENV_FILE=""
fi

usage() {
  cat >&2 <<'EOF'
usage: go2w_task_fsm.sh start        Launch the singleton task-FSM container
       go2w_task_fsm.sh status       Report the task-FSM container state
       go2w_task_fsm.sh down         Cleanly stop and remove the task-FSM container
EOF
  exit 2
}

container_running() {
  [[ "$($DOCKER inspect "$CONTAINER" --format '{{.State.Running}}' 2>/dev/null || true)" == "true" ]]
}

container_exists() {
  $DOCKER inspect "$CONTAINER" >/dev/null 2>&1
}

# Ask the running FSM whether its status topic has a live publisher. Sourced
# through the container's own overlay so the query participant matches the node.
status_topic_has_publisher() {
  $DOCKER exec "$CONTAINER" bash -lc '
    source /opt/ros/jazzy/setup.bash >/dev/null 2>&1
    source /opt/z_manip_ws/install/setup.bash >/dev/null 2>&1
    export ROS_DOMAIN_ID='"$DOMAIN_ID"' RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    count="$(timeout 6 ros2 topic info '"$STATUS_TOPIC"' 2>/dev/null \
      | sed -n "s/^Publisher count: //p")"
    [[ "${count:-0}" =~ ^[0-9]+$ ]] && (( count >= 1 ))
  ' >/dev/null 2>&1
}

supervisor_reported_ready() {
  $DOCKER logs "$CONTAINER" 2>&1 | grep -Fq "$READY_MARKER"
}

start_fsm() {
  if container_running; then
    printf 'task FSM already running (container %s); use `manip status task` or `manip down`.\n' \
      "$CONTAINER"
    return 0
  fi
  # Clear any exited remnant so the name is free and old logs do not confuse
  # the readiness poll below.
  container_exists && $DOCKER rm -f "$CONTAINER" >/dev/null 2>&1 || true

  local run_args=(
    -d --name "$CONTAINER" --restart no --network host
    -e "ROS_DOMAIN_ID=$DOMAIN_ID"
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    -e CYCLONEDDS_URI=file:///config/cyclonedds.xml
    -e "Z_MANIP_STACK_CONFIG=$IN_STACK_CONFIG"
    -e "Z_MANIP_ROBOT_URDF=$IN_ROBOT_URDF"
    -e "Z_MANIP_ROBOT_DESCRIPTION_FILE=$IN_ROBOT_URDF"
    -e "Z_MANIP_COLLISION_MODEL_FILE=$IN_COLLISION_MODEL"
    -v "$DDS_CONFIG:/config/cyclonedds.xml:ro"
    -v "$ROBOT_ASSETS:/robot/assets:ro"
  )
  [[ -n "$ENV_FILE" && -f "$ENV_FILE" ]] && run_args+=(--env-file "$ENV_FILE")

  $DOCKER run "${run_args[@]}" "$IMAGE" \
    "$SUPERVISOR_BIN" \
    --namespace "$NAMESPACE" \
    --startup-timeout "$STARTUP_TIMEOUT" \
    --shutdown-timeout "$SHUTDOWN_TIMEOUT" \
    -- \
    use_sim_time:=false \
    "robot_description_file:=$IN_ROBOT_URDF" \
    "stack_config_path:=$IN_STACK_CONFIG" \
    "collision_model_file:=$IN_COLLISION_MODEL" \
    >/dev/null

  # Poll for the supervisor's own readiness marker. The supervisor fails closed
  # (exits non-zero) on a graph conflict or a readiness timeout, so a container
  # that has exited is a definite, legible failure -- surface its own words.
  local deadline=$((SECONDS + STARTUP_TIMEOUT + 30))
  while ((SECONDS < deadline)); do
    if supervisor_reported_ready; then
      printf 'Task FSM is uniquely ready (container %s, ROS_DOMAIN_ID=%s).\n' \
        "$CONTAINER" "$DOMAIN_ID"
      return 0
    fi
    if ! container_running; then
      printf 'Task FSM failed to come up. Supervisor said:\n' >&2
      $DOCKER logs --tail 25 "$CONTAINER" 2>&1 | sed 's/^/  /' >&2
      return 1
    fi
    sleep 1
  done
  printf 'Task FSM did not report ready within %ss; leaving it running for inspection.\n' \
    "$((STARTUP_TIMEOUT + 30))" >&2
  $DOCKER logs --tail 15 "$CONTAINER" 2>&1 | sed 's/^/  /' >&2
  return 1
}

status_fsm() {
  local state="offline" summary="task FSM container not running"
  if container_running; then
    if supervisor_reported_ready && status_topic_has_publisher; then
      state="healthy"
      summary="singleton task runtime ready; $STATUS_TOPIC has a publisher"
    elif supervisor_reported_ready; then
      state="degraded"
      summary="supervisor reported ready but $STATUS_TOPIC has no publisher"
    else
      state="degraded"
      summary="container up but supervisor has not reported unique readiness yet"
    fi
  elif container_exists; then
    summary="task FSM container exited: $($DOCKER inspect "$CONTAINER" \
      --format '{{.State.Status}} (exit {{.State.ExitCode}})' 2>/dev/null || echo unknown)"
  fi
  printf 'task\t%s\t%s\n' "$state" "$summary"
}

down_fsm() {
  if ! container_exists; then
    printf 'Task FSM is already stopped.\n'
    return 0
  fi
  # A clean `docker stop` sends SIGTERM to the supervisor, which tears down only
  # the launch process group it created and releases its deployment lock. This
  # touches nothing but the container this manager owns.
  $DOCKER stop --time "$SHUTDOWN_TIMEOUT" "$CONTAINER" >/dev/null 2>&1 || true
  $DOCKER rm -f "$CONTAINER" >/dev/null 2>&1 || true
  printf 'Task FSM stopped. Perception, UI, and base chain were left unchanged.\n'
}

# Allow tests to source this file for function-level checks without dispatching.
(return 0 2>/dev/null) && __sourced=1 || __sourced=0
if ((__sourced)); then
  return 0
fi

case "${1:-}" in
  start) [[ $# -eq 1 ]] || usage; start_fsm ;;
  status) [[ $# -eq 1 ]] || usage; status_fsm ;;
  down) [[ $# -eq 1 ]] || usage; down_fsm ;;
  *) usage ;;
esac
