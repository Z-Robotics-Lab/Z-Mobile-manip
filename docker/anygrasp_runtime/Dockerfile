FROM anygrasp:cu128-py311

RUN python -m pip install --no-cache-dir "pyzmq==26.4.0" "msgpack==1.1.0"

COPY docker/anygrasp_runtime/server.py /opt/z_manip_anygrasp/server.py
COPY docker/anygrasp_runtime/healthcheck.py /opt/z_manip_anygrasp/healthcheck.py

ENV ANYGRASP_PORT=5557 \
    ANYGRASP_CHECKPOINT=/opt/anygrasp/grasp_detection/log/checkpoint_detection.tar \
    PYTHONUNBUFFERED=1

EXPOSE 5557
HEALTHCHECK --interval=10s --timeout=5s --start-period=180s --retries=3 \
  CMD ["python", "/opt/z_manip_anygrasp/healthcheck.py"]

CMD ["python", "/opt/z_manip_anygrasp/server.py"]
