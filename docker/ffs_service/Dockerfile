# Fast-FoundationStereo inference environment.
#
# Reuse the already-deployed CUDA/PyTorch visual runtime (torch 2.8.0+cu128,
# python 3.11, CUDA 12.8) so we avoid a second multi-gigabyte CUDA pull and keep
# the Fast-FoundationStereo deployment deterministic offline -- the same pattern
# yoloe_service uses.  Fast-FFS's PyTorch inference path needs only pure-python
# deps on top (no flash-attn, no xformers), plus a C compiler because Torch/
# Triton JIT-compiles a launcher stub at first inference.
FROM z-manip-edgetam:latest

USER root
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# build-essential: Triton's runtime driver compiles a small CPU launcher module
# on first GPU kernel use; without a C compiler inference dies in torch._inductor.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY docker/ffs_service/requirements.txt /app/requirements.txt
RUN python -m pip install --requirement /app/requirements.txt

# Headless OpenCV so the container never needs X11/libxcb at import time
# (the Fast-FFS demo/util scripts import cv2 at module load).
RUN python -m pip uninstall --yes opencv-python opencv-contrib-python 2>/dev/null || true \
    && python -m pip install opencv-python-headless==4.11.0.86

# The Fast-FoundationStereo repo (code + weights, ~/Z-Robotics-Lab/third_party/
# Fast-FoundationStereo) is mounted at /ffs at runtime -- the checkpoint is a
# fully-serialized torch module that needs the repo importable, and mounting
# keeps model upgrades rebuild-free.  Device calibration is mounted at
# /config/ffs_calibration.json (repo: configs/ffs_d435_calibration.json).
COPY docker/ffs_service/ffs_depth_service.py /app/ffs_depth_service.py

EXPOSE 8773
HEALTHCHECK --interval=10s --timeout=2s --start-period=120s --retries=3 \
  CMD python -c "import json,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:8773/health',timeout=1)); assert d['ready']" || exit 1

# 23-36-37 (most accurate distilled variant) at 8 refinement iters: measured on
# this 4090D at 640x480 -- 59 ms p50 / 1.27 GB VRAM, floor plane-fit RMS 1.07 mm
# vs 2.2 mm raw D435, temporal p95 42 mm vs 105 mm (shootout 2026-07-23).
CMD ["python", "/app/ffs_depth_service.py"]
