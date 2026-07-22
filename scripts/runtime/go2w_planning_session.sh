#!/usr/bin/env bash
set -euo pipefail

# One-shot real perception -> passive joint synchronization -> offline planning.
# CAN is assumed to have passed the boot-time receive-only gate. Per-request
# snapshots use the unprivileged recv-only probe and verify zero kernel TX.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STACK_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd -- "$STACK_ROOT/.." && pwd)"
IMAGE="${Z_MANIP_RUNTIME_IMAGE:-z-manip-runtime:pinocchio}"
IK_BACKEND="${Z_MANIP_IK_BACKEND:-pinocchio}"
TASK_PACKAGE_CONTAINER="${Z_MANIP_TASK_PACKAGE_CONTAINER:-/opt/z_manip_ws/install/lib/python3.12/site-packages/z_manip_task}"
PLANNING_ONLY_SEARCH_TIMEOUT_S="${Z_MANIP_PLANNING_ONLY_SEARCH_TIMEOUT_S:-6}"
PLANNING_ONLY_SYMMETRY_SAMPLES="${Z_MANIP_PLANNING_ONLY_SYMMETRY_SAMPLES:-4}"
PLANNING_ONLY_MAX_HYPOTHESES="${Z_MANIP_PLANNING_ONLY_MAX_HYPOTHESES:-64}"
PLANNING_ONLY_MAX_FEASIBLE_PLANS="${Z_MANIP_PLANNING_ONLY_MAX_FEASIBLE_PLANS:-1}"
# A small supported-object prior remains useful, but the mobile robot must not
# buy that preference by sweeping the wrist over its centreline lidar.  The
# stronger side/overhead policy lives in grasp_plan and is applied before IK.
SUPPORT_APPROACH_PRIOR_WEIGHT="${Z_MANIP_SUPPORT_APPROACH_PRIOR_WEIGHT:-0.05}"
PASSIVE_CAPTURE_SECONDS="${Z_MANIP_PASSIVE_CAPTURE_SECONDS:-0.25}"
REMOTE_PASSIVE_PROBE="/usr/local/libexec/z-manip/piper_passive_probe.py"
REMOTE_PASSIVE_REPORT="/tmp/z-manip-passive-live.json"
NUC_HOST="${GO2W_NUC_HOST:-yusenzlabnuc@192.168.3.8}"
NUC_KEY="${GO2W_NUC_SSH_KEY:-$HOME/.ssh/id_ed25519_codex_nuc}"
DDS_CONFIG="${Z_MANIP_DDS_CONFIG:-$STACK_ROOT/docker/runtime/cyclonedds-go2w-pc.xml}"
CALIBRATION="${Z_MANIP_CAMERA_CALIBRATION:-$WORKSPACE_ROOT/artifacts/go2w_real/calibration/piper_wrist_camera_calibration.json}"
URDF="${Z_MANIP_ROBOT_URDF:-$WORKSPACE_ROOT/go2W_Sim/assets/urdf/go2w_sensored.urdf}"
ROBOT_ASSETS="$(cd -- "$(dirname -- "$URDF")/.." && pwd)"
CONTAINER_URDF="/robot_assets/urdf/$(basename -- "$URDF")"
RUN_ROOT="${Z_MANIP_PLANNING_RUN_ROOT:-$WORKSPACE_ROOT/artifacts/go2w_real/planning_sessions}"
INSTRUCTION="${1:-pick the small white USB power adapter with the red port by its white body}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
RUN_DIR="$RUN_ROOT/$RUN_ID"
PERCEPTION_DIR="$RUN_DIR/perception"
PLANNING_DIR="$RUN_DIR/planning"
JOINT_REPORT="$RUN_DIR/passive_joint_report.json"
SESSION_GATE="$RUN_DIR/session_gate.json"
BUNDLE="$RUN_DIR/debug_bundle.json"

for path in "$NUC_KEY" "$DDS_CONFIG" "$CALIBRATION" "$URDF"; do
  [[ -f "$path" ]] || { printf 'required file is missing: %s\n' "$path" >&2; exit 1; }
done
docker image inspect "$IMAGE" >/dev/null
mkdir -p "$PERCEPTION_DIR" "$PLANNING_DIR"

ssh_args=(-i "$NUC_KEY" -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=5 "$NUC_HOST")
perception_pid=""
cleanup() {
  if [[ -n "$perception_pid" ]] && kill -0 "$perception_pid" >/dev/null 2>&1; then
    kill "$perception_pid" >/dev/null 2>&1 || true
    wait "$perception_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

printf 'Starting synchronized read-only perception session: %s\n' "$RUN_DIR"
perception_rc=0
Z_MANIP_ARTIFACT_DIR="$PERCEPTION_DIR" Z_MANIP_REQUIRE_PASSIVE_WINDOW=1 \
  "$SCRIPT_DIR/go2w_perception_lab.sh" dry-run "$INSTRUCTION" \
  >"$RUN_DIR/perception.log" 2>&1 &
perception_pid=$!
probe_index=0
probe_rc=0
while kill -0 "$perception_pid" >/dev/null 2>&1; do
  probe_index=$((probe_index + 1))
  probe_log="$RUN_DIR/passive_probe_${probe_index}.log"
  if ! ssh "${ssh_args[@]}" \
      /usr/bin/python3 "$REMOTE_PASSIVE_PROBE" \
        --interface can0 \
        --duration "$PASSIVE_CAPTURE_SECONDS" \
        --output "$REMOTE_PASSIVE_REPORT" \
      >"$probe_log" 2>&1; then
    probe_rc=1
    break
  fi
  temporary_report="$PERCEPTION_DIR/.live_passive_joint_report.json.tmp"
  if ! ssh "${ssh_args[@]}" cat "$REMOTE_PASSIVE_REPORT" \
      >"$temporary_report"; then
    probe_rc=1
    break
  fi
  mv -f "$temporary_report" "$PERCEPTION_DIR/live_passive_joint_report.json"
done
wait "$perception_pid" || perception_rc=$?
perception_pid=""
cat "$RUN_DIR/perception.log"
if [[ "$probe_rc" -ne 0 ]]; then
  printf 'passive CAN gate failed; see passive probe logs\n' >&2
  exit 1
fi
if [[ -f "$PERCEPTION_DIR/selected_passive_joint_report.json" ]]; then
  cp "$PERCEPTION_DIR/selected_passive_joint_report.json" "$JOINT_REPORT"
else
  latest_live="$PERCEPTION_DIR/live_passive_joint_report.json"
  [[ -f "$latest_live" ]] || { printf 'no passive joint report was captured\n' >&2; exit 1; }
  cp "$latest_live" "$JOINT_REPORT"
fi

gate_rc=0
PYTHONPATH="$STACK_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  python3 "$SCRIPT_DIR/piper_planning_session_gate.py" \
    --perception-dir "$PERCEPTION_DIR" \
    --joint-report "$JOINT_REPORT" \
    --calibration "$CALIBRATION" \
    --urdf "$URDF" \
    --output "$SESSION_GATE" || gate_rc=$?

planning_rc=1
if [[ "$perception_rc" -eq 0 && "$gate_rc" -eq 0 ]]; then
  joints_csv="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["joints_csv"])' "$SESSION_GATE")"
  planning_joints_csv="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["planning_joints_csv"])' "$SESSION_GATE")"
  planning_rc=0
  docker run --rm --network none \
    -e "Z_MANIP_IK_BACKEND=$IK_BACKEND" \
    -v "$PERCEPTION_DIR:/session/perception:ro" \
    -v "$PLANNING_DIR:/session/planning" \
    -v "$CALIBRATION:/session/calibration.json:ro" \
    -v "$ROBOT_ASSETS:/robot_assets:ro" \
    -v "$STACK_ROOT/configs/go2w_piper.json:/opt/z_manip/configs/go2w_piper.json:ro" \
    -v "$STACK_ROOT/configs/piper_collision_capsules.json:/opt/z_manip/configs/piper_collision_capsules.json:ro" \
    -v "$STACK_ROOT/z_manip:/opt/z_manip/python/z_manip:ro" \
    -v "$STACK_ROOT/ros2/z_manip_task/z_manip_task:$TASK_PACKAGE_CONTAINER:ro" \
    -v "$SCRIPT_DIR/piper_planning_dry_run.py:/usr/local/bin/z-manip-piper-planning-dry-run:ro" \
    "$IMAGE" z-manip-piper-planning-dry-run \
      --artifacts /session/perception \
      --config /opt/z_manip/configs/go2w_piper.json \
      --urdf "$CONTAINER_URDF" \
      --joints="$joints_csv" \
      --planning-joints="$planning_joints_csv" \
      --search-timeout-s "$PLANNING_ONLY_SEARCH_TIMEOUT_S" \
      --symmetry-samples "$PLANNING_ONLY_SYMMETRY_SAMPLES" \
      --max-hypotheses "$PLANNING_ONLY_MAX_HYPOTHESES" \
      --max-feasible-plans "$PLANNING_ONLY_MAX_FEASIBLE_PLANS" \
      --support-approach-prior-weight "$SUPPORT_APPROACH_PRIOR_WEIGHT" \
      --scene-clearance-m 0.001 \
      --scene-point-radius-m 0.001 \
      --gripper-scene-radius-scale 0.60 \
      --camera-calibration /session/calibration.json \
      --output /session/planning || planning_rc=$?
fi

bundle_args=(
  --perception-dir "$PERCEPTION_DIR"
  --session-gate "$SESSION_GATE"
  --joint-report "$JOINT_REPORT"
  --calibration "$CALIBRATION"
  --urdf "$URDF"
  --output "$BUNDLE"
)
[[ -f "$PLANNING_DIR/planning_report.json" ]] && bundle_args+=(--planning-dir "$PLANNING_DIR")
PYTHONPATH="$STACK_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  python3 "$SCRIPT_DIR/go2w_debug_bundle.py" "${bundle_args[@]}"
python3 "$SCRIPT_DIR/go2w_debug_safety_gate.py" \
  --bundle "$BUNDLE" \
  --artifact-root "$WORKSPACE_ROOT/artifacts" \
  --joint-report "$JOINT_REPORT" \
  --output "$RUN_DIR/debug_bundle.safety-audit.json" || true
ln -sfn "$RUN_ID" "$RUN_ROOT/latest_attempt"

printf 'Session artifacts: %s\n' "$RUN_DIR"
printf 'Debug dashboard: %s --bundle %s\n' "$SCRIPT_DIR/go2w_debug_ui.sh" "$BUNDLE"
if [[ "$perception_rc" -ne 0 || "$gate_rc" -ne 0 || "$planning_rc" -ne 0 ]]; then
  printf 'Planning session blocked; inspect session_gate.json and debug bundle.\n' >&2
  exit 1
fi
ln -sfn "$RUN_ID" "$RUN_ROOT/latest"
printf 'Planning-only session passed; motion commands published: 0\n'
