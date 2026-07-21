FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --requirement /app/requirements.txt \
    && useradd --create-home --uid 10001 edgetam \
    && mkdir -p /models/huggingface \
    && chown -R edgetam:edgetam /app /models

COPY --chown=edgetam:edgetam server.py /app/server.py
USER edgetam

EXPOSE 8092
HEALTHCHECK --interval=10s --timeout=2s --start-period=10s --retries=3 \
  CMD python -c "import json,urllib.request; json.load(urllib.request.urlopen('http://127.0.0.1:8092/health', timeout=1))" || exit 1

CMD ["python", "/app/server.py"]
