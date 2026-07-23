# Reuse the already-deployed CUDA/PyTorch visual runtime. This avoids a second
# multi-gigabyte CUDA pull and keeps YOLOE deployment deterministic offline.
FROM z-manip-edgetam:latest

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    YOLOE_MODEL=/models/yoloe-11s-seg.pt

WORKDIR /app
COPY docker/yoloe_service/requirements.txt /app/requirements.txt
RUN python -m pip install --requirement /app/requirements.txt
# The service never opens GUI windows. Replace Ultralytics' default OpenCV
# wheel so the container does not require X11/libxcb at import time.
RUN python -m pip uninstall --yes opencv-python \
    && python -m pip install opencv-python-headless==4.11.0.86

# Resolve both model and text-prompt encoder while the image build has network
# access. The deployed loopback service itself runs with no network access.
RUN mkdir -p /models /tmp/ultralytics \
    && cd /models \
    && python -c "from ultralytics import YOLO; m=YOLO('yoloe-11s-seg.pt'); m.set_classes(['bottle'])"
# Ultralytics resolves the MobileCLIP asset relative to the process working
# directory on every set_classes() call. Expose the baked, read-only asset
# there so runtime startup never attempts a network download.
RUN ln -s /models/mobileclip_blt.ts /app/mobileclip_blt.ts

COPY scripts/runtime/local_grounding_service.py /app/local_grounding_service.py

EXPOSE 8771
HEALTHCHECK --interval=10s --timeout=2s --start-period=30s --retries=3 \
  CMD python -c "import json,urllib.request; d=json.load(urllib.request.urlopen('http://127.0.0.1:8771/health',timeout=1)); assert d['ready'] and d['backend']=='yoloe'" || exit 1

# imgsz 960 (was 640): 20-seed bag recert with the identical production
# GroundingRuntime filter lifts qualified seeds 13->14 @0.15 and 10->13 @0.20
# with 0 false positives, warm inference p95 14ms (<<60ms), and raises the 2m
# box-top white-charger frame score 0.195->0.262. The 11s weight stays the clear
# winner over 11m/26s/26m (26m@960 introduces a false positive; the others do not
# raise the qualified count). minimum-confidence stays 0.15: although the score
# distribution shifts upward, 0.15 maximizes qualified (14 vs 13 at 0.20) while
# holding false positives at 0, so the conservative choice is to leave it.
CMD ["python", "/app/local_grounding_service.py", "--host", "127.0.0.1", "--port", "8771", "--model", "/models/yoloe-11s-seg.pt", "--minimum-confidence", "0.15", "--imgsz", "960"]
