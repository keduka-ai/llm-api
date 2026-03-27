# Dockerfile for RunPod Serverless deployment
# Uses the official llama.cpp CUDA server image with a RunPod handler overlay.

FROM ghcr.io/ggml-org/llama.cpp:server-cuda

USER root

# Ensure llama.cpp shared libs (libmtmd.so etc.) are discoverable
RUN ldconfig /app 2>/dev/null; true
ENV LD_LIBRARY_PATH="/app:${LD_LIBRARY_PATH}" \
    PATH="/app:${PATH}"

# Build args — MODEL selects from the catalog in download-models.sh
# Use a catalog alias (e.g. "qwen3.5-9b") or a direct HTTPS URL to a GGUF file.
ARG MODEL=""

# Install Python and pip (the llama.cpp image is minimal, no Python included)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    wget \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install RunPod and HuggingFace Hub (pinned in requirements.txt)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create models directory
RUN mkdir -p /models

# Download model at build time via the catalog in download-models.sh.
# MODEL is a catalog alias (e.g. "qwen3.5-9b") or a direct HTTPS URL.
COPY model-defaults.sh download-models.sh /tmp/
RUN chmod +x /tmp/download-models.sh && \
    MODEL="$MODEL" MODELS_DIR=/models /tmp/download-models.sh

# Copy handler source, config, and entrypoint
COPY src/ /workspace/src/
COPY handler.py /workspace/handler.py
COPY config/ /workspace/config/
COPY model-defaults.sh /workspace/model-defaults.sh
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# Environment variables (see .env.example for full list)
ENV MODELS_DIR=/models \
    REASONING_FORMAT=deepseek \
    N_GPU_LAYERS=-1 \
    N_CTX=20000 \
    N_BATCH=512 \
    N_UBATCH=1024 \
    FLASH_ATTN_MODE=on \
    MAX_GENERATION_TOKENS=75000 \
    DEFAULT_MAX_TOKENS=4096 \
    MAX_MESSAGES=256

WORKDIR /workspace

EXPOSE 8080

ENTRYPOINT []
CMD ["/workspace/entrypoint.sh"]
