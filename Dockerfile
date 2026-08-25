# RiskLens - container image for the scoring API and analyst console.
#
# Why a multi-stage build
# -----------------------
# Stage 1 ("builder") installs dependencies, which needs compilers and pulls
# in build-time caches. Stage 2 copies ONLY the installed packages and the
# application code. The compilers and caches never reach the final image.
#
# That matters for two reasons:
#   * size    - a smaller image pulls faster and costs less to store
#   * security - a compiler in a production container is an attacker's tool.
#                Fewer packages means fewer CVEs to patch.

# =========================================================================
# Stage 1 - builder
# =========================================================================
FROM python:3.11-slim AS builder

# Build-only dependencies. These do NOT reach the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy ONLY the dependency manifests first.
#
# This is the most important line in the file for build speed. Docker caches
# each layer and invalidates it when its inputs change. Because requirements
# change far less often than source code, putting them in their own layer
# means editing a .py file does NOT trigger a full dependency reinstall.
COPY requirements.txt pyproject.toml ./

# CPU-only torch. The default wheel bundles CUDA and is ~2.5 GB; we do not
# have a GPU in this deployment and would never use it.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        torch==2.5.1 \
    && pip install --no-cache-dir -r requirements.txt

# =========================================================================
# Stage 2 - runtime
# =========================================================================
FROM python:3.11-slim AS runtime

# Only the runtime libraries. libgomp is required by XGBoost (OpenMP);
# omitting it produces a confusing import error at container start.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Never run as root. If the application is compromised, the attacker lands as
# an unprivileged user who cannot modify system files or install packages.
RUN useradd --create-home --shell /bin/bash risklens

WORKDIR /app

# Take the installed dependency tree from the builder.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Application code and configuration.
COPY --chown=risklens:risklens src/     ./src/
COPY --chown=risklens:risklens scripts/ ./scripts/
COPY --chown=risklens:risklens app/     ./app/
COPY --chown=risklens:risklens configs/ ./configs/
COPY --chown=risklens:risklens corpus/  ./corpus/
COPY --chown=risklens:risklens pyproject.toml README.md ./

# Model artefacts are NOT baked into the image. They are mounted at runtime
# (see docker-compose.yml). Two reasons:
#   * a retrained model should not require an image rebuild
#   * models can be large, and image layers are immutable and cached forever
RUN mkdir -p /app/models /app/indexes /app/reports /app/data \
    && chown -R risklens:risklens /app

USER risklens

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000 8501

# The healthcheck hits /health, which reports whether the MODEL loaded - not
# merely whether the process is alive. A container serving 503s because the
# model is missing should be reported unhealthy, not healthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        r=urllib.request.urlopen('http://localhost:8000/health',timeout=4); \
        sys.exit(0 if r.status==200 else 1)"

CMD ["uvicorn", "risklens.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
