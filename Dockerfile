# Dockerfile for RunPod Serverless deployment
# Provides LLM inference via llama-cpp-python with CUDA support

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Build args
ARG MODEL_URL=""

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip setuptools wheel

# Install llama-cpp-python with CUDA support
# Try upstream PyPI first (faster, pre-built wheels available for common CUDA versions),
# fall back to JamePeng fork built from source (supports newer model architectures).
ENV CMAKE_ARGS="-DGGML_CUDA=on"
ENV FORCE_CMAKE=1
RUN pip install --no-cache-dir 'llama-cpp-python[server]' \
    || pip install --no-cache-dir 'llama-cpp-python[server] @ git+https://github.com/JamePeng/llama-cpp-python.git'

# Install RunPod and HuggingFace Hub
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create models directory
RUN mkdir -p /models

# Download model at build time if MODEL_URL is provided
RUN if [ -n "$MODEL_URL" ]; then \
        python -c "from huggingface_hub import hf_hub_download; import os; \
url='$MODEL_URL'; \
parts=url.rstrip('/').replace('https://huggingface.co/', '').split('/'); \
repo_id='/'.join(parts[0:2]); \
filename='/'.join(parts[4:]) if len(parts) > 4 else parts[-1]; \
print(f'Downloading {filename} from {repo_id}'); \
hf_hub_download(repo_id=repo_id, filename=filename, local_dir='/models')"; \
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
