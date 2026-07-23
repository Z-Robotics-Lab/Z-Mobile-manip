#!/usr/bin/env python3
"""NUC-side IR pair throttle for the Fast-FoundationStereo depth path.

Why this exists: the D435's infra1/infra2 compressed streams at 30 fps consume
enough Wi-Fi airtime to collapse the color stream from ~30 Hz to ~6 Hz on the
PC (measured 2026-07-23).  The FFS relay only consumes ~10 pairs/s, so this
node runs ON THE NUC (loopback DDS cost only), joins the two compressed IR
streams by exact frameset stamp, and republishes the *paired* messages on
dedicated topics at a bounded rate.  Only these throttled topics are
subscribed across Wi-Fi.

    in :  /nuc/camera/infra1/image_rect_raw/compressed   (30 fps, LAN-local)
          /nuc/camera/infra2/image_rect_raw/compressed
    out:  /nuc/camera/ffs_ir_pair/infra1/compressed      (<= FFS_IR_PAIR_HZ)
          /nuc/camera/ffs_ir_pair/infra2/compressed      (same stamps, paired)

Headers (stamp + frame_id) pass through untouched, preserving the exact-stamp
contract the downstream relay and perception node depend on.

Deployed at ~/ffs_ir_pair_throttle.py on the NUC, run by the user systemd unit
ffs-ir-throttle.service (repo source: configs/ffs-ir-throttle.service).
"""
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

RATE_HZ = float(os.environ.get('FFS_IR_PAIR_HZ', '10'))
IN1 = os.environ.get('FFS_IR_IN1', '/nuc/camera/infra1/image_rect_raw/compressed')
IN2 = os.environ.get('FFS_IR_IN2', '/nuc/camera/infra2/image_rect_raw/compressed')
OUT1 = os.environ.get('FFS_IR_OUT1', '/nuc/camera/ffs_ir_pair/infra1/compressed')
OUT2 = os.environ.get('FFS_IR_OUT2', '/nuc/camera/ffs_ir_pair/infra2/compressed')
BUFFER = 30


class IrPairThrottle(Node):
    def __init__(self):
        super().__init__('ffs_ir_pair_throttle')
        self._b1 = {}
        self._b2 = {}
        self._last_pub = 0.0
        self._interval = 1.0 / RATE_HZ if RATE_HZ > 0 else 0.0
        self._pub1 = self.create_publisher(CompressedImage, OUT1,
                                           qos_profile_sensor_data)
        self._pub2 = self.create_publisher(CompressedImage, OUT2,
                                           qos_profile_sensor_data)
        self.create_subscription(CompressedImage, IN1,
                                 lambda m: self._cb(m, self._b1),
                                 qos_profile_sensor_data)
        self.create_subscription(CompressedImage, IN2,
                                 lambda m: self._cb(m, self._b2),
                                 qos_profile_sensor_data)
        self.get_logger().info(
            f'IR pair throttle up: {IN1}+{IN2} -> {OUT1}+{OUT2} @<={RATE_HZ}Hz')

    def _cb(self, msg: CompressedImage, store: dict):
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        store[stamp] = msg
        while len(store) > BUFFER:
            store.pop(min(store))
        other = self._b2 if store is self._b1 else self._b1
        if stamp not in other:
            return
        m1 = self._b1.pop(stamp)
        m2 = self._b2.pop(stamp)
        now = time.monotonic()
        if now - self._last_pub < self._interval:
            return                      # rate cap: drop this frameset
        self._last_pub = now
        self._pub1.publish(m1)
        self._pub2.publish(m2)


def main():
    rclpy.init()
    node = IrPairThrottle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
