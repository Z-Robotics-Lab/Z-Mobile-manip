#!/usr/bin/env bash
set -euo pipefail

# Decode the NUC's bandwidth-efficient image_transport streams back to the
# canonical raw RGB-D contract consumed by Z-Manip.  This bridge subscribes
# permanently instead of using image_transport republish's matched-reader
# callback, which is intentionally lazy and can stop upstream transport during
# DDS endpoint rediscovery.  Keeping NUC sources under /nuc also prevents raw
# images from crossing Wi-Fi.
exec ros2 run z_manip_rgbd_bridge nonlazy_rgbd_bridge
