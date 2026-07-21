#!/usr/bin/env bash
set -eo pipefail

source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
source /opt/z_manip_ws/install/setup.bash
set -u

command -v curl >/dev/null
command -v z-manip-mobile-manipulation >/dev/null
command -v z-manip-mobile-manipulation-acceptance >/dev/null

python3 - <<'PY'
import importlib

modules = (
    "z_manip",
    "z_manip.perception.vlm_affordance",
    "z_manip_ros.contract",
    "z_manip_ros.vlm_edgetam_bridge",
    "z_manip_motion.contracts",
    "z_manip_motion.moveit_plan_bridge",
    "z_manip_edgetam.core",
    "z_manip_edgetam.node",
    "z_manip_task.core",
    "z_manip_task.planning",
    "z_manip_task.node",
    "z_manip_navigation.core",
    "z_manip_navigation.node",
    "z_manip_place.core",
    "z_manip_place.node",
    "vision_msgs.msg",
    "moveit_msgs.msg",
    "moveit_msgs.srv",
)
for module in modules:
    importlib.import_module(module)
print(f"python imports: {len(modules)}/{len(modules)}")
PY

packages=(
    z_manip_ros
    z_manip_motion
    z_manip_edgetam
    z_manip_task
    z_manip_navigation
    z_manip_place
    vision_msgs
    moveit_msgs
    moveit_kinematics
    moveit_planners_ompl
    moveit_ros_move_group
    moveit_ros_occupancy_map_monitor
    moveit_ros_perception
    rosbag2_storage_mcap
    trac_ik_kinematics_plugin
)
for package in "${packages[@]}"; do
    ros2 pkg prefix "${package}" >/dev/null
done
echo "ROS packages: ${#packages[@]}/${#packages[@]}"

ros2 pkg executables z_manip_ros | grep -qx \
    'z_manip_ros vlm_edgetam_bridge'
ros2 pkg executables z_manip_motion | grep -qx \
    'z_manip_motion motion_plan_bridge'
ros2 pkg executables z_manip_edgetam | grep -qx \
    'z_manip_edgetam edgetam_adapter'
ros2 pkg executables z_manip_task | grep -qx \
    'z_manip_task mobile_manipulation_runtime'
ros2 pkg executables z_manip_navigation | grep -qx \
    'z_manip_navigation coarse_navigation'
ros2 pkg executables z_manip_place | grep -qx \
    'z_manip_place observed_placement'
ros2 launch z_manip_ros perception.launch.py --show-args >/dev/null
ros2 launch z_manip_motion moveit_planning.launch.py --show-args >/dev/null
ros2 launch z_manip_edgetam edgetam.launch.py --show-args >/dev/null
ros2 launch z_manip_task task_runtime.launch.py --show-args >/dev/null
ros2 launch z_manip_navigation coarse_navigation.launch.py --show-args >/dev/null
ros2 launch z_manip_place place.launch.py --show-args >/dev/null
ros2 launch z_manip_task mobile_manipulation.launch.py --show-args >/dev/null
echo "launch descriptions: 7/7"

if [[ "${1:-}" == "--node-probe" ]]; then
    motion_log="$(mktemp)"
    perception_log="$(mktemp)"
    ros2 run z_manip_motion motion_plan_bridge \
        --ros-args -p use_sim_time:=false >"${motion_log}" 2>&1 &
    motion_pid=$!
    ros2 launch z_manip_ros perception.launch.py \
        start_edge_tam:=false use_sim_time:=false \
        >"${perception_log}" 2>&1 &
    perception_pid=$!

    cleanup() {
        kill "${motion_pid}" "${perception_pid}" 2>/dev/null || true
        wait "${motion_pid}" 2>/dev/null || true
        wait "${perception_pid}" 2>/dev/null || true
        rm -f "${motion_log}" "${perception_log}"
    }
    trap cleanup EXIT

    probe_node() {
        local node_name="$1"
        local process_id="$2"
        local log_file="$3"
        for _ in $(seq 1 30); do
            if ros2 node list 2>/dev/null | grep -qx "${node_name}"; then
                echo "node probe ready: ${node_name}"
                return
            fi
            if ! kill -0 "${process_id}" 2>/dev/null; then
                cat "${log_file}" >&2
                echo "node process exited: ${node_name}" >&2
                exit 1
            fi
            sleep 0.2
        done
        cat "${log_file}" >&2
        echo "node was not discoverable: ${node_name}" >&2
        exit 1
    }

    probe_node '/z_manip_motion_plan_bridge' "${motion_pid}" "${motion_log}"
    probe_node '/vlm_edgetam_bridge' "${perception_pid}" "${perception_log}"
fi

if [[ "${1:-}" != "--build-check" ]]; then
    echo "runtime smoke: PASS"
fi
