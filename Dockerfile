# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12
ARG BASE_IMAGE=python:${PYTHON_VERSION}-slim-bookworm
ARG UV_VERSION=0.11.15

FROM ${BASE_IMAGE} AS builder

ARG UV_VERSION
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential git \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install "uv==${UV_VERSION}"

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev --no-editable

FROM builder AS development-builder
RUN uv sync --frozen --extra dev --no-editable

FROM ${BASE_IMAGE} AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/var/cache/text2motion/huggingface \
    TORCH_HOME=/var/cache/text2motion/torch \
    XDG_CACHE_HOME=/var/cache/text2motion \
    VIRTUAL_ENV=/app/.venv \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        ffmpeg \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 text2motion \
    && useradd --uid 10001 --gid text2motion --create-home text2motion \
    && mkdir -p /app/checkpoints /app/data /app/logs /app/outputs /var/cache/text2motion \
    && chown -R text2motion:text2motion /app /var/cache/text2motion

COPY --from=builder --chown=text2motion:text2motion /app/.venv /app/.venv
COPY --chown=text2motion:text2motion configs ./configs

USER text2motion
ENTRYPOINT ["text2motion"]
CMD ["--help"]

FROM runtime AS development

COPY --from=development-builder --chown=text2motion:text2motion /app/.venv /app/.venv
COPY --chown=text2motion:text2motion tests ./tests
COPY --chown=text2motion:text2motion scripts ./scripts

USER text2motion
ENTRYPOINT ["python", "-m", "pytest"]
CMD ["-q"]
