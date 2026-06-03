# Dockerfile for RunPod Serverless deployment of Gemma 4 E2B-it.
# Uses the official llama.cpp CUDA server image with a RunPod handler overlay.

FROM ghcr.io/ggml-org/llama.cpp:server-cuda

USER root

# Ensure llama.cpp shared libs (libmtmd.so etc.) are discoverable.
RUN ldconfig /app 2>/dev/null; true
ENV LD_LIBRARY_PATH="/app:${LD_LIBRARY_PATH}" \
    PATH="/app:${PATH}"

# MODEL selects from the catalog in download-models.sh: a catalog alias
# (e.g. "gemma-4-e2b-it") or a direct HTTPS URL to a GGUF file.
ARG MODEL="gemma-4-e2b-it"

# System deps (the llama.cpp image is minimal).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, reproducible Python installs.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv venv --python 3.11 /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV="/opt/venv"

# Runtime Python deps (runpod, huggingface_hub) — pinned in requirements.txt.
COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --no-cache -r /tmp/requirements.txt

# Download the model at build time (baked in → cold start is load-only).
RUN mkdir -p /models
COPY model-defaults.sh download-models.sh /tmp/
RUN chmod +x /tmp/download-models.sh && \
    MODEL="$MODEL" MODELS_DIR=/models /tmp/download-models.sh

# Copy handler source, config, and entrypoint.
COPY src/ /workspace/src/
COPY handler.py /workspace/handler.py
COPY config/ /workspace/config/
COPY model-defaults.sh /workspace/model-defaults.sh
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# Runtime defaults — kept in sync with src/handler.py and config/__init__.py.
# (See README for the full env-var reference.)
ENV MODELS_DIR=/models \
    N_GPU_LAYERS=-1 \
    N_CTX=40192 \
    N_BATCH=512 \
    N_UBATCH=1024 \
    FLASH_ATTN_MODE=on \
    LLAMA_HEALTH_TIMEOUT=300 \
    DEFAULT_MODEL_NAME=gemma-4-e2b-it \
    MAX_GENERATION_TOKENS=40192 \
    DEFAULT_MAX_TOKENS=4096 \
    MAX_MESSAGES=256 \
    MAX_CONTENT_LENGTH=500000 \
    MAX_STOP_SEQUENCES=16

WORKDIR /workspace

EXPOSE 8080

ENTRYPOINT []
CMD ["/workspace/entrypoint.sh"]
