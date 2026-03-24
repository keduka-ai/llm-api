# Dockerfile for RunPod Serverless deployment
# Uses the official llama.cpp CUDA server image with a RunPod handler overlay.

FROM ghcr.io/ggml-org/llama.cpp:server-cuda

USER root

# Ensure llama.cpp shared libs (libmtmd.so etc.) are discoverable
RUN ldconfig /app 2>/dev/null; true
ENV LD_LIBRARY_PATH="/app:${LD_LIBRARY_PATH}"

# Build args
ARG MODEL_URL=""

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

# Download model at build time using download script or MODEL_URL fallback.
# MODEL_URL is validated: only https:// allowed, filename sanitised.
COPY download-models.sh /tmp/download-models.sh
RUN chmod +x /tmp/download-models.sh && \
    if [ -n "$MODEL_URL" ]; then \
        case "$MODEL_URL" in https://*) ;; *) echo "ERROR: MODEL_URL must use https://" && exit 1;; esac && \
        FILENAME=$(basename "$MODEL_URL" | sed 's/?.*//' | tr -cd 'A-Za-z0-9._-') && \
        if [ -z "$FILENAME" ]; then echo "ERROR: could not derive filename from MODEL_URL" && exit 1; fi && \
        echo "Downloading ${FILENAME} to /models/" && \
        wget -q --show-progress -O "/models/${FILENAME}" "$MODEL_URL"; \
    else \
        MODELS_DIR=/models /tmp/download-models.sh; \
    fi

# Copy handler source and entrypoint
COPY src/ /workspace/src/
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# Environment variables (see .env.example for full list)
# RUNPOD_MODE controls which model is loaded: "instruct" or "reasoning"
ENV RUNPOD_MODE=instruct \
    MODELS_DIR=/models \
    INSTRUCT_MODEL=Qwen3.5-4B-Q4_1.gguf \
    REASONING_MODEL=Phi-4-mini-reasoning-UD-Q8_K_XL.gguf \
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
