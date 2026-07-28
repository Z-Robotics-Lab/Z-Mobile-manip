# syntax=docker/dockerfile:1.7
#
# Hermetic VAMP (Vector-Accelerated Motion Planning) image.
#
# Why this exists: VAMP has no binary wheel on PyPI (sdist only), and its CMake
# build fetches CPM, nanobind, nigh, pdqsort and SIMDxorshift over the network at
# *configure* time. `scripts/offline/planning_replay_benchmark.py` runs with
# `--network none`, so a source build can never happen at run time. This image
# does all networked work in the builder stage and ships a self-contained wheel,
# preserving the offline harness's `--network none` property.
#
# Build:
#   docker build -t zmm/vamp:0.6.4 -f docker/vamp/Dockerfile docker/vamp
# Prove hermeticity:
#   docker run --rm --network none zmm/vamp:0.6.4 \
#       python3 -c "import vamp; print(vamp.robots)"
#
# CPU requirement: AVX2 (x86-64) or NEON (aarch64). VAMP will not build or run
# without SIMD. The workstation this was validated on is an i7-13700KF (avx2+bmi2).

# ---------------------------------------------------------------------------
# Stage 1: builder -- has network, compiles the wheel.
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Ubuntu 24.04 ships python3.12, matching the cp312 target of the VAMP wheel and
# the host workstation's interpreter.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        g++ \
        git \
        libeigen3-dev \
        ninja-build \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Pin the VAMP release that this toolchain was validated against. Bump
# deliberately: the generated robot headers are tied to the VAMP C++ API.
ARG VAMP_VERSION=0.6.4

# CPM_SOURCE_CACHE keeps the fetched dependencies inside the build context so a
# rebuild with a warm cache does not re-hit the network.
ENV CPM_SOURCE_CACHE=/opt/cpm_cache

WORKDIR /build

# Build the wheel once, here, where network is available.
RUN python3 -m venv /opt/buildvenv \
    && /opt/buildvenv/bin/pip install --no-cache-dir --upgrade pip wheel setuptools \
    && /opt/buildvenv/bin/pip wheel --no-cache-dir --wheel-dir /build/wheels \
        "vamp-planner==${VAMP_VERSION}" \
    && ls -la /build/wheels

# ---------------------------------------------------------------------------
# Stage 2: runtime -- no network needed, ever.
# ---------------------------------------------------------------------------
FROM ubuntu:24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/wheels /opt/wheels

# --no-index guarantees the install cannot reach out to PyPI: everything must be
# satisfiable from the wheels copied out of the builder stage.
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --no-index \
        --find-links /opt/wheels vamp-planner \
    && /opt/venv/bin/python -c "import vamp; print('vamp robots:', vamp.robots)"

ENV PATH=/opt/venv/bin:${PATH} \
    VIRTUAL_ENV=/opt/venv

# Self-check at build time so a broken image fails loudly here rather than in the
# replay harness.
RUN python3 -c "import vamp, numpy; \
    e = vamp.Environment(); \
    r_min, r_max = vamp.ur5.min_max_radii(); \
    pts = numpy.zeros((16, 3), dtype=numpy.float32); \
    e.add_pointcloud(pts.tolist(), r_min, r_max, vamp.POINT_RADIUS); \
    assert vamp.ur5.validate([0.0] * 6, e) in (True, False); \
    print('VAMP hermetic self-check OK')"

WORKDIR /work

CMD ["python3", "-c", "import vamp; print('vamp robots:', vamp.robots)"]
