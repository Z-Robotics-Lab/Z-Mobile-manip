#!/usr/bin/env python3
"""Fast-FoundationStereo depth service.

Loopback HTTP service (EdgeTAM/YOLOE pattern) that turns a rectified D435
left/right IR pair into metric depth registered in the COLOR optical frame.

  POST /infer  body JSON:
      {"ir1": <b64 png/jpeg bytes>, "ir2": <b64 png/jpeg bytes>}
    -> {"ok": true, "width": 640, "height": 480,
        "depth_zlib": <b64 zlib(uint16 mm, row-major)>,
        "timings_ms": {...}}
  GET /health -> {"ready": true, "model": ..., "iters": ...}

Design notes:
- The model checkpoint is a fully-serialized torch module; the
  Fast-FoundationStereo repo must be importable (mounted at /ffs).
- Registration IR1->COLOR runs on the GPU (project, transform, z-buffer
  scatter with a 2x2 splat).  Validated against librealsense rs.align:
  median |dz| = 1.09 mm, p95 10.2 mm, coverage parity (reg_validate 2026-07-23).
- The left-occlusion band (u - disparity < 0) is marked invalid before
  registration, matching the FFS repo's remove_invisible recommendation, so
  hallucinated matches never become collision geometry.
- Calibration is the factory calibration of the specific wrist D435i, loaded
  from /config/ffs_calibration.json (configs/ffs_d435_calibration.json in the
  repo).  The ROS relay cross-checks live camera_info and refuses to feed us
  if the device does not match.
"""
import base64
import json
import logging
import os
import sys
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch

FFS_DIR = os.environ.get('FFS_DIR', '/ffs')
sys.path.insert(0, FFS_DIR)
from core.utils.utils import InputPadder  # noqa: E402

MODEL_PATH = os.environ.get(
    'FFS_MODEL_DIR', '/ffs/weights/23-36-37/model_best_bp2_serialize.pth')
CALIB_PATH = os.environ.get('FFS_CALIB', '/config/ffs_calibration.json')
ITERS = int(os.environ.get('FFS_ITERS', '8'))
MAX_DISP = int(os.environ.get('FFS_MAX_DISP', '192'))
PORT = int(os.environ.get('FFS_PORT', '8773'))
ZMAX_M = float(os.environ.get('FFS_ZMAX_M', '10.0'))
AMP_DTYPE = torch.float16

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger('ffs_depth_service')


class Engine:
    def __init__(self):
        with open(CALIB_PATH) as f:
            c = json.load(f)
        self.calib = c
        self.W, self.H = int(c['width']), int(c['height'])
        ir = c['K_ir1']
        self.fx_ir, self.fy_ir = float(ir['fx']), float(ir['fy'])
        self.cx_ir, self.cy_ir = float(ir['ppx']), float(ir['ppy'])
        kc = c['K_color']
        self.fx_c, self.fy_c = float(kc['fx']), float(kc['fy'])
        self.cx_c, self.cy_c = float(kc['ppx']), float(kc['ppy'])
        self.baseline = float(c['baseline_m'])
        dev = torch.device('cuda')
        # rs.extrinsics rotation is column-major.
        R = np.array(c['ir1_to_color']['rotation_colmajor'],
                     dtype=np.float64).reshape(3, 3, order='F')
        t = np.array(c['ir1_to_color']['translation_m'], dtype=np.float64)
        self.R = torch.as_tensor(R, dtype=torch.float32, device=dev)
        self.t = torch.as_tensor(t, dtype=torch.float32, device=dev)
        us, vs = np.meshgrid(np.arange(self.W), np.arange(self.H))
        self.us = torch.as_tensor(us.reshape(-1), dtype=torch.float32, device=dev)
        self.vs = torch.as_tensor(vs.reshape(-1), dtype=torch.float32, device=dev)

        torch.autograd.set_grad_enabled(False)
        log.info('loading model %s (iters=%d)', MODEL_PATH, ITERS)
        self.model = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
        self.model.args.valid_iters = ITERS
        self.model.args.max_disp = MAX_DISP
        self.model.cuda().eval()
        self.lock = threading.Lock()
        # Warm up: first forward JIT-compiles Triton kernels (seconds).
        dummy = np.zeros((self.H, self.W), np.uint8)
        self._disparity(dummy, dummy)
        self._disparity(dummy, dummy)
        log.info('engine warm; ready')

    def _disparity(self, ir1: np.ndarray, ir2: np.ndarray) -> torch.Tensor:
        a = np.repeat(ir1[..., None], 3, axis=2)
        b = np.repeat(ir2[..., None], 3, axis=2)
        t0 = torch.as_tensor(a).cuda().float()[None].permute(0, 3, 1, 2)
        t1 = torch.as_tensor(b).cuda().float()[None].permute(0, 3, 1, 2)
        padder = InputPadder(t0.shape, divis_by=32, force_square=False)
        t0, t1 = padder.pad(t0, t1)
        with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
            disp = self.model.forward(
                t0, t1, iters=ITERS, test_mode=True,
                optimize_build_volume='pytorch1')
        return padder.unpad(disp.float()).reshape(self.H, self.W)

    def _register_to_color(self, depth_ir: torch.Tensor) -> torch.Tensor:
        """GPU z-buffer reprojection IR1 frame -> COLOR frame (2x2 splat)."""
        z = depth_ir.reshape(-1)
        ok = z > 0.05
        z = z[ok]
        if z.numel() == 0:
            return torch.zeros((self.H, self.W), device=depth_ir.device)
        x = (self.us[ok] - self.cx_ir) * z / self.fx_ir
        y = (self.vs[ok] - self.cy_ir) * z / self.fy_ir
        P = torch.stack([x, y, z], 0)
        Pc = self.R @ P + self.t[:, None]
        zc = Pc[2]
        g = zc > 0.05
        uf = self.fx_c * Pc[0] / zc + self.cx_c
        vf = self.fy_c * Pc[1] / zc + self.cy_c
        buf = torch.zeros(self.H * self.W, device=z.device)
        idxs, zs = [], []
        for du in (0, 1):
            for dv in (0, 1):
                uc = torch.floor(uf).long() + du
                vc = torch.floor(vf).long() + dv
                inb = g & (uc >= 0) & (uc < self.W) & (vc >= 0) & (vc < self.H)
                idxs.append(vc[inb] * self.W + uc[inb])
                zs.append(zc[inb])
        idx = torch.cat(idxs)
        zz = torch.cat(zs)
        order = torch.argsort(zz, descending=True)  # nearest written last
        buf[idx[order]] = zz[order]
        return buf.reshape(self.H, self.W)

    def infer(self, ir1_bytes: bytes, ir2_bytes: bytes) -> dict:
        """GPU-thread-affine entry point.

        MUST run on the single persistent worker thread (see _gpu_worker):
        PyTorch cuDNN/cuBLAS handles and kernel caches are thread-local, so
        calling CUDA from a fresh thread per HTTP request costs ~900 ms of
        per-thread re-initialisation (measured live 2026-07-23: 916 ms p50
        from ThreadingHTTPServer threads vs 45 ms on a stable thread).
        """
        t0 = time.time()
        ir1 = cv2.imdecode(np.frombuffer(ir1_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        ir2 = cv2.imdecode(np.frombuffer(ir2_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if ir1 is None or ir2 is None:
            raise ValueError('IR image decode failed')
        if ir1.shape != (self.H, self.W) or ir2.shape != (self.H, self.W):
            raise ValueError(f'unexpected IR shape {ir1.shape}/{ir2.shape}')
        t1 = time.time()
        with self.lock:
            disp = self._disparity(ir1, ir2)
            # Left-occlusion band: pixels whose match falls left of the right
            # image edge were never observed by the right camera.
            u = self.us.reshape(self.H, self.W)
            disp = torch.where(u - disp < 0, torch.zeros_like(disp), disp)
            disp = disp.clamp(min=0.0)
            depth_ir = torch.where(
                disp > 1e-2,
                self.fx_ir * self.baseline / disp.clamp(min=1e-6),
                torch.zeros_like(disp))
            depth_ir = torch.where(depth_ir < ZMAX_M, depth_ir,
                                   torch.zeros_like(depth_ir))
            depth_color = self._register_to_color(depth_ir)
            torch.cuda.synchronize()
        t2 = time.time()
        mm = (depth_color * 1000.0).clamp(0, 65535).to(torch.int32) \
            .cpu().numpy().astype(np.uint16)
        payload = base64.b64encode(zlib.compress(mm.tobytes(), 1)).decode()
        t3 = time.time()
        return dict(ok=True, width=self.W, height=self.H,
                    depth_zlib=payload,
                    valid_frac=float((mm > 0).mean()),
                    timings_ms=dict(decode=(t1 - t0) * 1e3,
                                    gpu=(t2 - t1) * 1e3,
                                    encode=(t3 - t2) * 1e3,
                                    total=(t3 - t0) * 1e3))


ENGINE = None


class GpuWorker:
    """Single persistent GPU thread; HTTP threads hand work over and wait."""

    def __init__(self, engine: Engine):
        self._engine = engine
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._job = None          # (ir1, ir2, reply_box, reply_event)
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name='ffs-gpu-worker')
        self._thread.start()

    def submit(self, ir1: bytes, ir2: bytes, timeout: float = 10.0) -> dict:
        box = {}
        done = threading.Event()
        with self._cond:
            # Latest-only: an unstarted queued job is superseded, and its
            # waiter is released with an explicit busy error.
            if self._job is not None:
                old = self._job
                old[2]['error'] = 'superseded'
                old[3].set()
            self._job = (ir1, ir2, box, done)
            self._cond.notify()
        if not done.wait(timeout):
            return dict(ok=False, error='gpu worker timeout')
        if 'error' in box:
            return dict(ok=False, error=box['error'])
        return box['result']

    def _run(self):
        while True:
            with self._cond:
                while self._job is None:
                    self._cond.wait()
                ir1, ir2, box, done = self._job
                self._job = None
            try:
                box['result'] = self._engine.infer(ir1, ir2)
            except Exception as exc:  # noqa: BLE001
                log.exception('infer failed')
                box['error'] = str(exc)
            done.set()


WORKER = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == '/health':
            self._send(200, dict(ready=ENGINE is not None,
                                 model=MODEL_PATH, iters=ITERS,
                                 serial=ENGINE.calib.get('serial') if ENGINE else None))
        else:
            self._send(404, dict(ok=False))

    def do_POST(self):
        if self.path != '/infer':
            self._send(404, dict(ok=False))
            return
        try:
            n = int(self.headers.get('Content-Length', '0'))
            req = json.loads(self.rfile.read(n))
            out = WORKER.submit(base64.b64decode(req['ir1']),
                                base64.b64decode(req['ir2']))
            self._send(200 if out.get('ok') else 500, out)
        except Exception as exc:  # noqa: BLE001
            log.exception('request failed')
            self._send(500, dict(ok=False, error=str(exc)))


def main():
    global ENGINE, WORKER
    ENGINE = Engine()
    WORKER = GpuWorker(ENGINE)
    # Warm the worker thread itself: its first CUDA call pays the one-time
    # thread-local cuDNN/cuBLAS handle initialisation.
    blank = cv2.imencode('.png', np.zeros((ENGINE.H, ENGINE.W), np.uint8))[1].tobytes()
    out = WORKER.submit(blank, blank, timeout=120.0)
    log.info('worker warm: %s', out.get('timings_ms'))
    srv = ThreadingHTTPServer(('0.0.0.0', PORT), Handler)
    log.info('serving on :%d', PORT)
    srv.serve_forever()


if __name__ == '__main__':
    main()
