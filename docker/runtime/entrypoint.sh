#!/usr/bin/env bash
set -eo pipefail

source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
source /opt/z_manip_ws/install/setup.bash
set -u

exec "$@"
