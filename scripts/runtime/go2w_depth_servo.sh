#!/usr/bin/env bash
set -euo pipefail

# Fixed launcher used by the loopback UI.  The browser can choose only the
# shadow/live mode; topic names, gains, bounds, ROS domain, and executable are
# owned here rather than accepted as arbitrary HTTP input.
if (($# != 2)); then
  printf 'usage: %s {shadow|live} ABSOLUTE_STATUS_PATH\n' "$0" >&2
  exit 2
fi

MODE="$1"
STATUS_PATH="$2"
TRACE_PATH="${STATUS_PATH%.json}.trace.jsonl"
RUNTIME_STATE_PATH="$(dirname -- "$STATUS_PATH")/runtime-observer.json"
case "$MODE" in
  shadow|live) ;;
  *) printf 'mode must be shadow or live\n' >&2; exit 2 ;;
esac
if [[ "$STATUS_PATH" != /* ]]; then
  printf 'status path must be absolute\n' >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STACK_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
DDS_CONFIG="$STACK_ROOT/docker/runtime/cyclonedds-go2w-pc.xml"
RUNTIME_IMAGE="${Z_MANIP_WHOLE_BODY_IMAGE:-z-mobile-manip-whole-body:latest}"
WHOLE_BODY_URDF="${Z_MANIP_WHOLE_BODY_URDF:-$STACK_ROOT/../go2W_Sim/assets/urdf/go2w_sensored.urdf}"
WHOLE_BODY_CALIBRATION="${Z_MANIP_WHOLE_BODY_CALIBRATION:-$STACK_ROOT/../artifacts/go2w_real/calibration/piper_wrist_camera_calibration.json}"
WHOLE_BODY_COLLISION_MODEL="${Z_MANIP_WHOLE_BODY_COLLISION_MODEL:-$STACK_ROOT/configs/piper_collision_capsules.json}"
CONTAINER_NAME="z-manip-go2w-depth-servo"
NUC_HOST="${GO2W_NUC_HOST:-yusenzlabnuc@192.168.3.8}"
NUC_KEY="${GO2W_NUC_SSH_KEY:-$HOME/.ssh/id_ed25519_codex_nuc}"
ARM_OWNER_STARTED=0

release_arm_owner() {
  if [[ "$ARM_OWNER_STARTED" == 1 ]]; then
    if ! ssh -i "$NUC_KEY" -o BatchMode=yes -o ConnectTimeout=5 "$NUC_HOST" \
      'set -eu; systemctl --user stop z-mobile-manip-piper-reactive-view.service; systemctl --user restart z-manip-piper-passive-feedback.service; systemctl --user is-active --quiet z-manip-piper-passive-feedback.service' \
      >/dev/null 2>&1; then
      printf 'warning: failed to restore the passive PiPER feedback owner on the NUC\n' >&2
    fi
    ARM_OWNER_STARTED=0
  fi
}

acquire_arm_owner() {
  # The reactive service conflicts with the passive listener, so systemd stops
  # the old CAN owner and starts the new one in one transaction. If start-up or
  # the postcondition check fails, restore the passive owner before returning.
  ssh -i "$NUC_KEY" -o BatchMode=yes -o ConnectTimeout=5 "$NUC_HOST" '
    set -eu
    restore_passive() {
      systemctl --user stop z-mobile-manip-piper-reactive-view.service >/dev/null 2>&1 || true
      systemctl --user restart z-manip-piper-passive-feedback.service
    }
    # H5.  NOTHING IN THIS REPOSITORY EVER CALLED reset-failed.  A unit that hit
    # its start limiter stays in failed/start-limit-hit until something clears
    # it, and the ONLY thing an operator saw was `systemctl start` returning
    # non-zero under `set -eu`, which aborts this whole launcher.  "The arm
    # owner could not be acquired" and "systemd is refusing to try" are not the
    # same fault and must not look the same.
    #
    # This is a DIAGNOSIS first and a clear second.  The clear is scoped to
    # start-limit-hit only, happens at most once per launcher invocation, and
    # does NOT weaken anything: the is-active postcondition below is unchanged,
    # so a unit that is genuinely broken still fails this function and still
    # stops the servo.  What changes is that the operator is told why.
    for unit in z-mobile-manip-piper-reactive-view.service z-manip-piper-passive-feedback.service; do
      result="$(systemctl --user show "$unit" -p Result --value 2>/dev/null || true)"
      if [ "$result" = start-limit-hit ]; then
        restarts="$(systemctl --user show "$unit" -p NRestarts --value 2>/dev/null || true)"
        printf "%s tripped its systemd start limiter (Result=start-limit-hit, NRestarts=%s); clearing it once so this launch reports the REAL fault instead of systemd refusing to try\n" \
          "$unit" "${restarts:-unknown}" >&2
        systemctl --user reset-failed "$unit" >/dev/null 2>&1 || true
      fi
    done
    if ! systemctl --user start z-mobile-manip-piper-reactive-view.service; then
      restore_passive
      exit 1
    fi
    if ! systemctl --user is-active --quiet z-mobile-manip-piper-reactive-view.service ||
       systemctl --user is-active --quiet z-manip-piper-passive-feedback.service; then
      restore_passive
      exit 1
    fi
  '
  ARM_OWNER_STARTED=1
}

cleanup() {
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  release_arm_owner
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -r "$DDS_CONFIG" ]]; then
  printf 'missing PC CycloneDDS profile: %s\n' "$DDS_CONFIG" >&2
  exit 1
fi
if ! command -v docker >/dev/null 2>&1; then
  printf 'docker is required for the fixed CycloneDDS depth-servo runtime\n' >&2
  exit 1
fi
[[ -r "$WHOLE_BODY_URDF" ]] || {
  printf 'missing whole-body URDF: %s\n' "$WHOLE_BODY_URDF" >&2
  exit 1
}
[[ -r "$WHOLE_BODY_CALIBRATION" ]] || {
  printf 'missing measured hand-eye calibration: %s\n' "$WHOLE_BODY_CALIBRATION" >&2
  exit 1
}
[[ -r "$WHOLE_BODY_COLLISION_MODEL" ]] || {
  printf 'missing whole-body collision model: %s\n' "$WHOLE_BODY_COLLISION_MODEL" >&2
  exit 1
}
if ! docker image inspect "$RUNTIME_IMAGE" >/dev/null 2>&1; then
  printf 'building one-time CasADi whole-body runtime image: %s\n' "$RUNTIME_IMAGE" >&2
  docker build -t "$RUNTIME_IMAGE" \
    -f "$STACK_ROOT/docker/whole_body_runtime/Dockerfile" "$STACK_ROOT"
fi

# THE DEPLOYMENT IS THE WORKING TREE.  The `-v "$STACK_ROOT:$STACK_ROOT:ro"`
# line in the docker run below mounts this checkout INTO the container and the
# servo runs straight out of it, so editing the checkout while a task is live
# is a deploy to a powered-up robot.  Recorded
# consequences: one session's traceback holds _tick at 1835, 1954, 2043, 2045
# and 2183 -- five different programs inside one run -- and commits a0f9e10
# (10:38) and 0124a20 (12:10) each produced a resident-worker fingerprint
# mismatch minutes later, a component self-heal, 900-1200 passive-window
# rejections and PERCEPTION_PROCESS_FAILED on a plainly detected object, 2 of 2,
# while the same sparse target succeeded 5 times with 0 rejections.
#
# WHAT I RECOMMEND AND DID NOT DO: bind-mount a CONTENT-ADDRESSED COPY of
# $STACK_ROOT (rsync into artifacts/go2w_real/.deploy/<fingerprint>/ and mount
# that instead).  It is the only option that makes a running task genuinely
# immune to an edit, and it also makes the traceback line numbers meaningful
# again.  It is not implemented here because it rewrites the launch path of a
# robot this change cannot be executed against, and a mistake in it does not
# degrade the servo, it prevents the servo from starting at all.
#
# WHAT IS IMPLEMENTED: the other option in the pair -- refuse to launch on a
# dirty tree -- DOWNGRADED TO A LOUD WARNING CARRYING THE FINGERPRINT, and said
# so here.  Refusing would be wrong for this operator: the tree is dirty for
# most of a working session by design (that is how the servo gets new code at
# all), so a refusal would block them mid-session on the normal case and teach
# them to bypass the check.  The warning is unconditional and the fingerprint it
# prints is the same value the servo stamps into depth-servo.json and every
# trace row, so an after-the-fact analysis can prove which bytes ran.
#
# THE FINGERPRINT MUST BE MEASURABLE OVER THE SAME FILES ON BOTH SIDES OF THE
# CONTAINER BOUNDARY, AND ONCE IT WAS NOT.  runtime_inputs() hashes one file
# that lives OUTSIDE $STACK_ROOT ($WORKSPACE_ROOT/go2W_Sim/.../go2w_sensored.urdf).
# The docker run below mounts $STACK_ROOT at its own path but mounted that URDF
# only at /robot/go2w_sensored.urdf, so the servo's in-container re-measure saw
# 98 files where this launcher saw 99, differed by construction, and reported
# runtime_fingerprint_mutated: true on every trace row of every run against a
# frozen tree.  --launch-manifest returns the digest AND every hashed path
# outside $STACK_ROOT; each is mounted below at its own absolute path so the
# two measurements enumerate the same set.  ONE python3 call, so the digest and
# the mount list cannot describe different measurements.
RUNTIME_FINGERPRINT=""
FINGERPRINT_INPUT_MOUNTS=()
while IFS= read -r fingerprint_input; do
  [[ -n "$fingerprint_input" ]] || continue
  if [[ -z "$RUNTIME_FINGERPRINT" ]]; then
    RUNTIME_FINGERPRINT="$fingerprint_input"
  else
    FINGERPRINT_INPUT_MOUNTS+=(-v "$fingerprint_input:$fingerprint_input:ro")
  fi
done < <(
  /usr/bin/env python3 "$SCRIPT_DIR/z_manip_runtime_fingerprint.py" \
    --launch-manifest 2>/dev/null || true
)
if [[ -z "$RUNTIME_FINGERPRINT" ]]; then
  # Not fatal: the fingerprint is diagnostic, not a gate.  But say so, because
  # an absent stamp reads identically to "checked and clean" downstream.
  printf 'warning: could not measure the runtime fingerprint; depth-servo.json and its trace will report unverified\n' >&2
fi
printf 'z_manip_runtime_fingerprint=%s\n' "${RUNTIME_FINGERPRINT:-unknown}" >&2
if git -C "$STACK_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  RUNTIME_TREE_STATUS="$(git -C "$STACK_ROOT" status --porcelain 2>/dev/null || true)"
  RUNTIME_TREE_COMMIT="$(git -C "$STACK_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
  if [[ -n "$RUNTIME_TREE_STATUS" ]]; then
    printf '\n' >&2
    printf '!! LIVE DEPLOY WARNING: %s is DIRTY at commit %s.\n' \
      "$STACK_ROOT" "$RUNTIME_TREE_COMMIT" >&2
    printf '!! This launcher bind-mounts that tree read-only and runs the servo from it.\n' >&2
    printf '!! Editing it during this task IS a deploy to a powered-up robot, and the\n' >&2
    printf '!! resident perception/planning workers will self-heal on the fingerprint change.\n' >&2
    printf '!! runtime fingerprint: %s\n' "${RUNTIME_FINGERPRINT:-unknown}" >&2
    printf '!! modified paths:\n' >&2
    printf '%s\n' "$RUNTIME_TREE_STATUS" | sed 's/^/!!   /' >&2
    printf '\n' >&2
  fi
fi

# A user service can remain active after Unitree's WebRTC data channel has
# closed. In that state ROS accepts /cmd_vel but the robot never receives it.
# Refuse to start a live servo until the fixed NUC transport has either been
# verified or restarted and reconnected. Shadow mode remains transport-free.
if [[ "$MODE" == live ]]; then
  "$SCRIPT_DIR/go2w_base_transport_preflight.sh"
  [[ -f "$NUC_KEY" ]] || { printf 'missing NUC SSH key: %s\n' "$NUC_KEY" >&2; exit 1; }
  acquire_arm_owner
fi

# The workstation host currently has FastDDS only, while every long-running
# Z-Manip ROS process and the NUC use CycloneDDS with explicit Wi-Fi peers.
# Running this small publisher in the runtime image makes discovery
# deterministic across the PC/NUC boundary.  A host FastDDS publisher can be
# visible to local containers yet disappear from the NUC, causing the base
# watchdog to stop a valid approach after the first short motion.
# EVERY freshness/loss-stair budget below is pinned explicitly, including the
# ones that already match the Python defaults.  A stair budget this file does
# not name is one an operator reading this launcher cannot see: that is how
# --transform-timeout-s (now --geometry-staleness-timeout-s) shipped at an
# invisible 0.25 s under a visible 0.40 s --target-timeout-s, so two dropped
# camera frames hard-zeroed the base with a reason string blaming TF.  The
# mapping is enforced from go2w_depth_servo.py's STAIR_SETTING_FLAGS by
# tests/test_go2w_depth_servo_runtime.py; adding a stair knob without pinning
# it here fails the suite.
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
docker run --rm \
  --name "$CONTAINER_NAME" \
  --network host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e ROS_DOMAIN_ID=20 \
  -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
  -e CYCLONEDDS_URI=file:///config/cyclonedds.xml \
  -e Z_MANIP_RUNTIME_FINGERPRINT="$RUNTIME_FINGERPRINT" \
  -e PYTHONPATH="$STACK_ROOT:/opt/z_manip/python" \
  -v "$DDS_CONFIG:/config/cyclonedds.xml:ro" \
  -v "$STACK_ROOT:$STACK_ROOT:ro" \
  -v "$WHOLE_BODY_URDF:/robot/go2w_sensored.urdf:ro" \
  ${FINGERPRINT_INPUT_MOUNTS[@]+"${FINGERPRINT_INPUT_MOUNTS[@]}"} \
  -v "$WHOLE_BODY_CALIBRATION:/robot/piper_wrist_camera_calibration.json:ro" \
  -v "$WHOLE_BODY_COLLISION_MODEL:/robot/piper_collision_capsules.json:ro" \
  -v "$(dirname -- "$STATUS_PATH"):$(dirname -- "$STATUS_PATH"):rw" \
  "$RUNTIME_IMAGE" \
  python3 "$SCRIPT_DIR/go2w_depth_servo.py" \
  --mode "$MODE" \
  --status-file "$STATUS_PATH" \
  --trace-file "$TRACE_PATH" \
  --target-topic /track_3d/selected_target_pointcloud \
  --tracking-topic /track_3d/is_tracking \
  --velocity-topic /cmd_vel \
  --runtime-state "$RUNTIME_STATE_PATH" \
  --runtime-transform-timeout-s 0.50 \
  --whole-body casadi \
  --whole-body-urdf /robot/go2w_sensored.urdf \
  --whole-body-calibration /robot/piper_wrist_camera_calibration.json \
  --whole-body-collision-model /robot/piper_collision_capsules.json \
  --desired-depth-m 0.50 \
  --handoff-depth-m 0.52 \
  --handoff-bearing-deg 20 \
  --min-forward-mps 0.10 \
  --max-forward-mps 0.18 \
  --max-yaw-rps 0.12 \
  --yaw-deadband-deg 10 \
  --max-yaw-step-rps 0.015 \
  --target-timeout-s 0.40 \
  --max-target-capture-age-s 0.70 \
  --tracking-hold-s 0.80 \
  --tracking-loss-grace-s 2.75 \
  --handoff-settle-s 0.30 \
  --geometry-staleness-timeout-s 0.40 \
  --rate-hz 20
