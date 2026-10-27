# Multi-stage production build for Universal Speech Gateway
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install a pinned uv for reproducible, lockfile-driven installs
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Dependencies first (cached layer), resolved exactly from uv.lock
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --extra all --no-install-project

# Then the project itself
COPY gateway/ ./gateway/
RUN uv sync --frozen --no-dev --extra all --no-editable

# Final runtime image
FROM python:3.11-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SPEECH_CONFIG_PATH="/app/config.yaml" \
    TTS_CONFIG_PATH="/app/config.yaml"

COPY gateway/ /app/gateway/
COPY config.yaml /app/config.yaml
COPY scripts/ /app/scripts/

# Create model directories
RUN mkdir -p /app/models/kokoro /app/models/whisper

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
