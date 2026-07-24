#!/usr/bin/env python3
"""Relay D435 IR pairs through the Fast-FoundationStereo depth service.

Runs in the z-manip-runtime image (host network, CycloneDDS domain 20) next to
the RGB-D bridge.  Subscribes the NUC's compressed infra1/infra2 streams (the
raw IR pair never crosses Wi-Fi uncompressed), exact-stamp pairs them, sends the
newest pair to the loopback FFS service, and publishes the returned COLOR-frame
depth as the drop-in replacement for /camera/aligned_depth_to_color/image_raw:

    /camera/ffs_depth_aligned/image_raw   sensor_msgs/Image 16UC1 (mm)
        header.stamp     = the IR pair stamp.  With enable_sync:=true the D435
                           publishes color/infra1/infra2/aligned_depth with
                           IDENTICAL stamps per frameset (verified live
                           121/121 exact matches), so downstream exact-stamp
                           color<->depth joins keep working unchanged.
        header.frame_id  = camera_color_optical_frame

Fail-closed calibration guard: on startup the relay compares the live infra1 +
color camera_info against configs/ffs_d435_calibration.json (mounted at
/config/ffs_calibration.json).  Any mismatch > 0.5 px or a serial change stops
the relay before a single depth frame is published.

Latest-only processing: the service round trip (~70-90 ms) is slower than the
30 fps camera, so pending pairs collapse to the freshest; published rate is
whatever the GPU sustains (measured and logged every 10 s).
"""
import base64
import json
import os
import sys
import threading
import time
import urllib.request

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

# Lightweight depth-image noise filter (pure numpy/cv2, no ROS/torch).  Mounted
# next to this script inside the relay container (see ffs_depth_stack.sh) and
# alongside it in the repo, so the import resolves both in-container and on-host
# for tests.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ffs_depth_filter import FilterConfig, filter_depth  # noqa: E402

SERVICE_URL = os.environ.get('FFS_SERVICE_URL', 'http://127.0.0.1:8773')
CALIB_PATH = os.environ.get('FFS_CALIB', '/config/ffs_calibration.json')
IR1_TOPIC = os.environ.get(
    'FFS_IR1_TOPIC', '/nuc/camera/infra1/image_rect_raw/compressed')
IR2_TOPIC = os.environ.get(
    'FFS_IR2_TOPIC', '/nuc/camera/infra2/image_rect_raw/compressed')
IR1_INFO_TOPIC = os.environ.get(
    'FFS_IR1_INFO_TOPIC', '/nuc/camera/infra1/camera_info')
COLOR_INFO_TOPIC = os.environ.get(
    'FFS_COLOR_INFO_TOPIC', '/nuc/camera/color/camera_info')
OUT_TOPIC = os.environ.get('FFS_OUT_TOPIC', '/camera/ffs_depth_aligned/image_raw')
PAIR_BUFFER = 60          # per-stream stamp buffer (2 s @ 30 fps)
CALIB_TOL_PX = 0.5
# GPU-contention throttle.  The EdgeTAM tracker shares this 4090; running FFS
# flat-out (~17 fps, ~90% GPU duty) starved EdgeTAM mask inference down to
# ~3.7 Hz (measured live 2026-07-23).  The perception node consumes depth at
# well below 10 Hz, so capping FFS here restores EdgeTAM throughput without
# costing the planning path anything.
MAX_FPS = float(os.environ.get('FFS_MAX_FPS', '10'))


class FfsDepthRelay(Node):
    def __init__(self):
        super().__init__('ffs_depth_relay')
        with open(CALIB_PATH) as f:
            self.calib = json.load(f)
        self.W = int(self.calib['width'])
        self.H = int(self.calib['height'])
        self.frame_id = self.calib['color_frame_id']

        self._ir1 = {}
        self._ir2 = {}
        self._latest_pair = None
        self._cond = threading.Condition()
        self._calib_ok = {'ir1': False, 'color': False}
        self._calib_failed = False
        self._published = 0
        self._received = 0
        self._bytes_in = 0
        self._lat_ms = []
        # Depth noise filter: applied here (the single publisher upstream of
        # every consumer) so one pass benefits EdgeTAM depth, grasp scene
        # points, the UI cloud and collision checking.  Disable at runtime with
        # FFS_FILTER=0 (restart, no rebuild -- relay + filter are bind-mounted).
        self._filter_cfg = FilterConfig.from_env()
        self._filter_prev = None
        self._filter_removed = 0
        self.get_logger().info(
            f'depth filter: enabled={self._filter_cfg.enabled} '
            f'stages={self._filter_cfg.active_stages()}')

        self._pub = self.create_publisher(Image, OUT_TOPIC,
                                          qos_profile_sensor_data)
        self.create_subscription(CompressedImage, IR1_TOPIC,
                                 lambda m: self._ir_cb(m, self._ir1),
                                 qos_profile_sensor_data)
        self.create_subscription(CompressedImage, IR2_TOPIC,
                                 lambda m: self._ir_cb(m, self._ir2),
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, IR1_INFO_TOPIC,
                                 lambda m: self._info_cb(m, 'ir1', 'K_ir1'),
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, COLOR_INFO_TOPIC,
                                 lambda m: self._info_cb(m, 'color', 'K_color'),
                                 qos_profile_sensor_data)
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()
        self.create_timer(10.0, self._report)
        self.get_logger().info(
            f'relay up: {IR1_TOPIC} + {IR2_TOPIC} -> {SERVICE_URL} -> {OUT_TOPIC}')

    # -- calibration guard ---------------------------------------------------
    def _info_cb(self, msg: CameraInfo, key: str, calib_key: str):
        if self._calib_ok[key] or self._calib_failed:
            return
        ref = self.calib[calib_key]
        live = dict(fx=msg.k[0], fy=msg.k[4], ppx=msg.k[2], ppy=msg.k[5])
        errs = {k: abs(live[k] - float(ref[k])) for k in live}
        if max(errs.values()) > CALIB_TOL_PX:
            self._calib_failed = True
            self.get_logger().fatal(
                f'{key} camera_info mismatch vs ffs_calibration.json: {errs} '
                '(camera replaced/recalibrated?). Relay will NOT publish; '
                're-run the calibration capture probe.')
            return
        self._calib_ok[key] = True
        self.get_logger().info(f'{key} camera_info matches calibration '
                               f'(max err {max(errs.values()):.3f}px)')

    # -- pairing -------------------------------------------------------------
    def _ir_cb(self, msg: CompressedImage, store: dict):
        stamp = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        self._bytes_in += len(msg.data)
        store[stamp] = bytes(msg.data)
        while len(store) > PAIR_BUFFER:
            store.pop(min(store))
        other = self._ir2 if store is self._ir1 else self._ir1
        if stamp in other:
            a = self._ir1.pop(stamp)
            b = self._ir2.pop(stamp)
            self._received += 1
            with self._cond:
                self._latest_pair = (stamp, a, b)   # latest-only
                self._cond.notify()

    # -- service round trip --------------------------------------------------
    def _loop(self):
        import zlib
        min_interval = 1.0 / MAX_FPS if MAX_FPS > 0 else 0.0
        last_start = 0.0
        while rclpy.ok():
            wait_s = min_interval - (time.time() - last_start)
            if wait_s > 0:
                time.sleep(wait_s)   # pairs collapse to latest meanwhile
            with self._cond:
                while self._latest_pair is None:
                    self._cond.wait(timeout=1.0)
                    if not rclpy.ok():
                        return
                stamp, a, b = self._latest_pair
                self._latest_pair = None
            last_start = time.time()
            if self._calib_failed:
                continue
            if not (self._calib_ok['ir1'] and self._calib_ok['color']):
                continue  # wait until both camera_infos verified
            t0 = time.time()
            try:
                body = json.dumps(dict(
                    ir1=base64.b64encode(a).decode(),
                    ir2=base64.b64encode(b).decode())).encode()
                req = urllib.request.Request(
                    SERVICE_URL + '/infer', data=body,
                    headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    out = json.loads(resp.read())
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warning(f'FFS service call failed: {exc}',
                                          throttle_duration_sec=5.0)
                continue
            if not out.get('ok'):
                self.get_logger().warning(f'FFS service error: {out.get("error")}',
                                          throttle_duration_sec=5.0)
                continue
            raw = zlib.decompress(base64.b64decode(out['depth_zlib']))
            if len(raw) != self.W * self.H * 2:
                self.get_logger().error('bad depth payload size; dropping')
                continue
            if self._filter_cfg.enabled:
                depth = np.frombuffer(raw, dtype=np.uint16).reshape(self.H, self.W)
                depth, report = filter_depth(depth, self._filter_cfg,
                                             self._filter_prev)
                self._filter_prev = depth
                self._filter_removed += report['removed']
                raw = np.ascontiguousarray(depth).tobytes()
            msg = Image()
            msg.header.stamp.sec = stamp // 10**9
            msg.header.stamp.nanosec = stamp % 10**9
            msg.header.frame_id = self.frame_id
            msg.height = self.H
            msg.width = self.W
            msg.encoding = '16UC1'
            msg.is_bigendian = False
            msg.step = self.W * 2
            msg.data = raw
            self._pub.publish(msg)
            self._published += 1
            self._lat_ms.append((time.time() - t0) * 1e3)

    def _report(self):
        lat = self._lat_ms[-100:]
        p50 = sorted(lat)[len(lat) // 2] if lat else float('nan')
        filt = ''
        if self._filter_cfg.enabled and self._published:
            filt = f' filt_drop~{self._filter_removed / self._published:.0f}px/frame'
        self.get_logger().info(
            f'pairs_in={self._received} published={self._published} '
            f'rate~{self._published / max(1e-9, 10.0):.1f}fps '
            f'rt_p50={p50:.0f}ms wifi_in={self._bytes_in / 10.0 / 1e6:.2f}MB/s'
            f'{filt}')
        self._published = 0
        self._bytes_in = 0
        self._filter_removed = 0
        if len(self._lat_ms) > 400:
            self._lat_ms = self._lat_ms[-100:]


def main():
    rclpy.init()
    node = FfsDepthRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
