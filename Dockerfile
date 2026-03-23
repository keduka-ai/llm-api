# Dockerfile for RunPod Serverless deployment
# Provides LLM inference via llama-cpp-python with CUDA support

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Build args
ARG MODEL_URL=""

# Install build dependencies (git needed for pip git+ installs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# Install llama-cpp-python with CUDA support using pre-built wheels.
# The cu124 index has wheels compiled for CUDA 12.4 — no source build needed.
# Falls back to source build from JamePeng fork if wheel doesn't support the model arch.
RUN pip install --no-cache-dir \
    'llama-cpp-python[server]' \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 \
    || (CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
        pip install --no-cache-dir \
        'llama-cpp-python[server] @ git+https://github.com/JamePeng/llama-cpp-python.git')

# Install RunPod and HuggingFace Hub
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create models directory
RUN mkdir -p /models

# Download model at build time if MODEL_URL is provided
# Supports direct download URLs (wget) — simplest and most reliable approach.
RUN if [ -n "$MODEL_URL" ]; then \
        FILENAME=$(basename "$MODEL_URL" | sed 's/?.*//')  && \
        echo "Downloading ${FILENAME} to /models/" && \
        wget -q --show-progress -O "/models/${FILENAME}" "$MODEL_URL"; \
    fi

# Copy handler source
COPY src/ /workspace/src/

# Environment variables
ENV RUNPOD_MODE=instruct \
    MODELS_DIR=/models \
    N_GPU_LAYERS=-1 \
    FLASH_ATTN=1

WORKDIR /workspace

CMD ["python", "-u", "src/handler.py"]
