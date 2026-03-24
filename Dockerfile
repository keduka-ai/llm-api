# Dockerfile for RunPod Serverless deployment
# Provides LLM inference via llama-cpp-python with CUDA support

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Build args
ARG MODEL_URL=""

# Install build dependencies (git needed for pip git+ installs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

# Install llama-cpp-python with CUDA 12.4 support from official pre-built wheels.
# Source: https://abetlen.github.io/llama-cpp-python/whl/cu124/
# These wheels ship with GPU support baked in — no source compilation needed.
RUN pip install --no-cache-dir \
    'llama-cpp-python[server]==0.3.16' \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124

# Verify the CUDA-enabled shared library was installed (libggml-cuda.so).
# We cannot call llama_supports_gpu_offload() at build time because it loads
# libllama.so which links against libcuda.so.1 — a host driver lib that only
# exists at runtime on GPU machines, not during docker build.
RUN python -c "\
import pathlib, sys; \
site = pathlib.Path(sys.prefix) / 'lib'; \
cuda_libs = list(pathlib.Path('/usr/local/lib/python3.11/dist-packages/llama_cpp/lib').glob('*cuda*')); \
assert cuda_libs, 'No CUDA shared libs found in llama_cpp — wrong wheel installed'; \
print(f'OK: found CUDA libs: {[p.name for p in cuda_libs]}')"

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

# Create non-root user for runtime
RUN useradd -m appuser && chown -R appuser:appuser /models

# Copy handler source
COPY src/ /workspace/src/

# Ensure the CUDA runtime/compat libs are on the linker path so libllama.so
# can find libcuda.so.1 at container startup on GPU hosts.
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/compat:${LD_LIBRARY_PATH}

# Environment variables (see .env.example for full list)
ENV RUNPOD_MODE=instruct \
    MODELS_DIR=/models \
    N_GPU_LAYERS=-1 \
    FLASH_ATTN=1 \
    MAX_GENERATION_TOKENS=75000 \
    DEFAULT_MAX_TOKENS=4096 \
    MAX_MESSAGES=256

WORKDIR /workspace

USER appuser

CMD ["python", "-u", "src/handler.py"]
