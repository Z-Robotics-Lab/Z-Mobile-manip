# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE=navstack:ready@sha256:7c1f317d0bfe33fbdb5067bf0266fc70c273da23bcbd767ebf3c714405118407
FROM ${BASE_IMAGE}

ARG ROS_DISTRO=jazzy
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# Keep this list explicit: the image is usable with navstack:ready today and
# with a stock ROS Jazzy ros-base image when supplied through BASE_IMAGE.
RUN test "${ROS_DISTRO}" = "jazzy" \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        python3-colcon-common-extensions \
        python3-numpy \
        python3-opencv \
        python3-scipy \
        python3-yaml \
        python3-zmq \
        python3-msgpack \
        ros-${ROS_DISTRO}-ament-index-python \
        ros-${ROS_DISTRO}-cv-bridge \
        ros-${ROS_DISTRO}-diagnostic-msgs \
        ros-${ROS_DISTRO}-geometry-msgs \
        ros-${ROS_DISTRO}-image-transport-plugins \
        ros-${ROS_DISTRO}-launch \
        ros-${ROS_DISTRO}-launch-ros \
        ros-${ROS_DISTRO}-message-filters \
        ros-${ROS_DISTRO}-moveit-kinematics \
        ros-${ROS_DISTRO}-moveit-msgs \
        ros-${ROS_DISTRO}-moveit-planners-ompl \
        ros-${ROS_DISTRO}-moveit-ros-move-group \
        ros-${ROS_DISTRO}-moveit-ros-occupancy-map-monitor \
        ros-${ROS_DISTRO}-moveit-ros-perception \
        ros-${ROS_DISTRO}-nav-msgs \
        ros-${ROS_DISTRO}-hpp-fcl \
        ros-${ROS_DISTRO}-pinocchio \
        ros-${ROS_DISTRO}-rclpy \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
        ros-${ROS_DISTRO}-robot-state-publisher \
        ros-${ROS_DISTRO}-sensor-msgs \
        ros-${ROS_DISTRO}-sensor-msgs-py \
        ros-${ROS_DISTRO}-shape-msgs \
        ros-${ROS_DISTRO}-std-msgs \
        ros-${ROS_DISTRO}-std-srvs \
        ros-${ROS_DISTRO}-tf2-ros \
        ros-${ROS_DISTRO}-topic-tools \
        ros-${ROS_DISTRO}-trac-ik-kinematics-plugin \
        ros-${ROS_DISTRO}-trajectory-msgs \
        ros-${ROS_DISTRO}-vision-msgs \
        ros-${ROS_DISTRO}-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*

# The core is intentionally copied without refs, models, evidence, or secrets.
# It remains a normal importable package while avoiding a system pip mutation.
WORKDIR /opt/z_manip
COPY z_manip/ /opt/z_manip/python/z_manip/
COPY configs/ /opt/z_manip/configs/
ENV PYTHONPATH=/opt/z_manip/python

WORKDIR /opt/z_manip_ws
COPY ros2/z_manip_ros/ src/z_manip_ros/
COPY ros2/z_manip_motion/ src/z_manip_motion/
COPY ros2/z_manip_edgetam/ src/z_manip_edgetam/
COPY ros2/z_manip_rgbd_bridge/ src/z_manip_rgbd_bridge/
COPY ros2/z_manip_task/ src/z_manip_task/
COPY ros2/z_manip_navigation/ src/z_manip_navigation/
COPY ros2/z_manip_place/ src/z_manip_place/

RUN source "/opt/ros/${ROS_DISTRO}/setup.bash" \
    && colcon build \
        --merge-install \
        --packages-select z_manip_ros z_manip_motion z_manip_edgetam \
            z_manip_rgbd_bridge \
            z_manip_navigation z_manip_place z_manip_task \
        --event-handlers console_direct+

COPY docker/runtime/entrypoint.sh /usr/local/bin/z-manip-entrypoint
COPY docker/runtime/smoke.sh /usr/local/bin/z-manip-runtime-smoke
COPY scripts/runtime/mobile_manipulation_supervisor.py \
    /usr/local/bin/z-manip-mobile-manipulation
COPY scripts/runtime/mobile_manipulation_acceptance.py \
    /usr/local/bin/z-manip-mobile-manipulation-acceptance
COPY scripts/runtime/real_rgbd_bridge.sh \
    /usr/local/bin/z-manip-real-rgbd-bridge
COPY scripts/runtime/go2w_perception_probe.py \
    /usr/local/bin/z-manip-go2w-perception-probe
COPY scripts/runtime/go2w_perception_dry_run.py \
    /usr/local/bin/z-manip-go2w-perception-dry-run
COPY scripts/runtime/go2w_perception_worker.py \
    /usr/local/bin/z-manip-go2w-perception-worker
COPY scripts/runtime/piper_planning_dry_run.py \
    /usr/local/bin/z-manip-piper-planning-dry-run
COPY scripts/runtime/piper_hand_eye_calibrate.py \
    /usr/local/bin/z-manip-piper-hand-eye-calibrate
COPY scripts/runtime/piper_hand_eye_sample.py \
    /usr/local/bin/z-manip-piper-hand-eye-sample
COPY scripts/runtime/piper_charuco_tool.py \
    /usr/local/bin/z-manip-piper-charuco
COPY scripts/runtime/go2w_latest_only_ros_verify.py \
    /usr/local/bin/z-manip-go2w-latest-only-verify
RUN chmod 0755 \
        /usr/local/bin/z-manip-entrypoint \
        /usr/local/bin/z-manip-runtime-smoke \
        /usr/local/bin/z-manip-mobile-manipulation \
        /usr/local/bin/z-manip-mobile-manipulation-acceptance \
        /usr/local/bin/z-manip-real-rgbd-bridge \
        /usr/local/bin/z-manip-go2w-perception-probe \
        /usr/local/bin/z-manip-go2w-perception-dry-run \
        /usr/local/bin/z-manip-go2w-perception-worker \
        /usr/local/bin/z-manip-piper-planning-dry-run \
        /usr/local/bin/z-manip-piper-hand-eye-calibrate \
        /usr/local/bin/z-manip-piper-hand-eye-sample \
        /usr/local/bin/z-manip-piper-charuco \
        /usr/local/bin/z-manip-go2w-latest-only-verify \
    && /usr/local/bin/z-manip-runtime-smoke --build-check \
    && dpkg-query -W \
        -f='${binary:Package}\t${Version}\t${Status}\n' \
        'ros-jazzy-moveit-*' \
        'ros-jazzy-hpp-fcl' \
        'ros-jazzy-pinocchio' \
        'ros-jazzy-trac-ik-*' \
        'ros-jazzy-vision-msgs' \
    | awk -F '\t' \
        '$3 == "install ok installed" {print $1 "\t" $2}' \
        > /opt/z_manip/runtime-package-manifest.txt

ENV DEBIAN_FRONTEND=
WORKDIR /opt/z_manip_ws
ENTRYPOINT ["/usr/local/bin/z-manip-entrypoint"]
CMD ["bash"]
