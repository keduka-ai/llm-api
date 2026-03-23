#!/bin/bash
set -euo pipefail

# =============================================================================
# RunPod Deploy Script
# Runs a single llama.cpp server (instruct OR reasoning) on a RunPod GPU pod.
# No docker compose required — installs llama-cpp-python with pre-built CUDA
# wheels and runs the OpenAI-compatible server. Falls back to building from
# source (latest llama.cpp) if the pre-built wheel doesn't support the model.
#
# USAGE:
#   1. Copy .env.example to .env and set RUNPOD_MODE:
#        cp .env.example .env
#        # Edit .env: set RUNPOD_MODE=instruct  or  RUNPOD_MODE=reasoning
#
#   2. Place your .gguf model file(s) in ai_api/models/ (or set MODELS_DIR).
#
#   3. Run the script:
#        bash runpod-deploy.sh
#
#      Or set the mode inline without a .env file:
#        RUNPOD_MODE=instruct  bash runpod-deploy.sh
#        RUNPOD_MODE=reasoning bash runpod-deploy.sh
#
#   4. The server exposes an OpenAI-compatible API:
#        POST http://<pod-ip>:8080/v1/chat/completions
#        GET  http://<pod-ip>:8080/health
#
#   5. To connect the Django API gateway, set in its environment:
#        MODEL1_NAME=instruct
#        MODEL1_URL=http://<pod-ip>:8080
#      (or MODEL2_NAME/MODEL2_URL for reasoning)
#
# OPTIONS (set via .env or environment):
#   RUNPOD_MODE          "instruct" or "reasoning" (default: instruct)
#   MODELS_DIR           Path to .gguf files (default: ./ai_api/models)
#   INSTRUCT_MODEL       Instruct .gguf filename (default: Qwen3.5-4B-Q4_1.gguf)
#   INSTRUCT_CTX_SIZE    Context window for instruct (default: 85000)
#   INSTRUCT_UBATCH_SIZE Micro-batch size for instruct (default: 2048)
#   REASONING_MODEL      Reasoning .gguf filename (default: Phi-4-mini-reasoning-UD-Q8_K_XL.gguf)
#   REASONING_CTX_SIZE   Context window for reasoning (default: 10000)
#   REASONING_UBATCH_SIZE Micro-batch size for reasoning (default: 1024)
#   LLAMA_PORT           Server listen port (default: 8080)
#   N_GPU_LAYERS         GPU offload layers (default: -1, all)
#   VENV_DIR             Virtual environment path (default: ./venv)
#   SKIP_VENV            Set to 1 to skip venv creation (default: 0)
#   CUDA_VERSION         CUDA version for pre-built wheels (default: auto-detected)
#   FORCE_BUILD          Set to 1 to skip wheels and build from source (default: 0)
#
# REQUIREMENTS:
#   - NVIDIA GPU with CUDA toolkit (RunPod GPU pods satisfy this)
#   - Python 3.10+ with venv module
#   - curl (for health checks)
#   - cmake, gcc/g++ (only if building from source)
#   - .gguf model file in MODELS_DIR
# =============================================================================

# --- Help flag ---
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    sed -n '5,/^# ====.*===$/p' "$0" | sed 's/^# \?//'
    exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load .env if present
if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
fi

# --- Detect CUDA version for pre-built wheels ---
detect_cuda_version() {
    if [ -n "${CUDA_VERSION:-}" ]; then
        echo "${CUDA_VERSION}"
        return
    fi

    if command -v nvcc &>/dev/null; then
        # e.g. "release 12.4" -> "cu124"
        local ver
        ver=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
        echo "cu${ver//./}"
        return
    fi

    if command -v nvidia-smi &>/dev/null; then
        local ver
        ver=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
        echo "cu${ver//./}"
        return
    fi

    echo ""
}

# --- Set up Python virtual environment ---
VENV_DIR="${VENV_DIR:-${SCRIPT_DIR}/venv}"
SKIP_VENV="${SKIP_VENV:-0}"

if [ "${SKIP_VENV}" != "1" ]; then
    if [ ! -d "${VENV_DIR}" ]; then
        echo "Creating virtual environment at ${VENV_DIR} (--system-site-packages)..."
        python3 -m venv --system-site-packages "${VENV_DIR}"
    fi

    echo "Activating virtual environment: ${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"

    # Install project requirements if not already satisfied
    REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements-latest.txt"
    if [ -f "${REQUIREMENTS_FILE}" ]; then
        echo "Installing requirements from ${REQUIREMENTS_FILE}..."
        pip install --quiet --no-cache-dir -r "${REQUIREMENTS_FILE}"
    fi
fi

# --- Install llama-cpp-python with CUDA support ---
FORCE_BUILD="${FORCE_BUILD:-0}"

install_from_wheel() {
    local cuda_ver
    cuda_ver=$(detect_cuda_version)

    if [ -n "${cuda_ver}" ]; then
        local wheel_url="https://abetlen.github.io/llama-cpp-python/whl/${cuda_ver}"
        echo "Installing llama-cpp-python[server] from pre-built CUDA wheels (${cuda_ver})..."
        pip install --no-cache-dir \
            'llama-cpp-python[server]' \
            --extra-index-url "${wheel_url}"
    else
        echo "WARNING: Could not detect CUDA version. Installing without GPU-specific wheels."
        echo "Set CUDA_VERSION (e.g. cu124) in .env to use pre-built CUDA wheels."
        pip install --no-cache-dir 'llama-cpp-python[server]'
    fi
}

install_from_source() {
    # The official PyPI llama-cpp-python (v0.3.16, Aug 2025) is too old for
    # newer architectures like Qwen3.5. JamePeng's fork tracks the latest
    # llama.cpp and adds support for these models.
    echo "Building llama-cpp-python from source with CUDA (JamePeng fork, latest llama.cpp)..."

    # Check build dependencies
    for cmd in cmake gcc g++; do
        if ! command -v "${cmd}" &>/dev/null; then
            echo "ERROR: ${cmd} is required to build from source."
            echo "Install with: apt-get update && apt-get install -y cmake build-essential"
            exit 1
        fi
    done

    if ! command -v nvcc &>/dev/null; then
        echo "ERROR: nvcc not found. CUDA toolkit is required for GPU support."
        exit 1
    fi

    # Uninstall old version first to avoid conflicts
    pip uninstall -y llama-cpp-python 2>/dev/null || true

    CMAKE_ARGS="-DGGML_CUDA=on" \
    FORCE_CMAKE=1 \
        pip install --no-cache-dir \
            'llama-cpp-python[server] @ git+https://github.com/JamePeng/llama-cpp-python.git'
}

if [ "${FORCE_BUILD}" = "1" ]; then
    install_from_source
elif ! python3 -c "import llama_cpp" &>/dev/null; then
    install_from_wheel
else
    echo "llama-cpp-python already installed."
fi

# --- Configuration from environment (with defaults) ---
RUNPOD_MODE="${RUNPOD_MODE:-instruct}"
MODELS_DIR="${MODELS_DIR:-${SCRIPT_DIR}/ai_api/models}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"

# --- Model-specific settings ---
case "${RUNPOD_MODE}" in
    instruct)
        MODEL_FILE="${INSTRUCT_MODEL:-Qwen3.5-4B-Q4_1.gguf}"
        MODEL_ALIAS="instruct"
        CTX_SIZE="${INSTRUCT_CTX_SIZE:-85000}"
        UBATCH_SIZE="${INSTRUCT_UBATCH_SIZE:-2048}"
        ;;
    reasoning)
        MODEL_FILE="${REASONING_MODEL:-Phi-4-mini-reasoning-UD-Q8_K_XL.gguf}"
        MODEL_ALIAS="reasoning"
        CTX_SIZE="${REASONING_CTX_SIZE:-10000}"
        UBATCH_SIZE="${REASONING_UBATCH_SIZE:-1024}"
        ;;
    *)
        echo "ERROR: RUNPOD_MODE must be 'instruct' or 'reasoning', got '${RUNPOD_MODE}'"
        exit 1
        ;;
esac

# Resolve MODELS_DIR to absolute path
MODELS_DIR="$(cd "${MODELS_DIR}" && pwd)"
MODEL_PATH="${MODELS_DIR}/${MODEL_FILE}"

# --- Validate model file exists ---
if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: Model file not found: ${MODEL_PATH}"
    echo "Available models in ${MODELS_DIR}:"
    ls -1 "${MODELS_DIR}"/*.gguf 2>/dev/null || echo "  (none)"
    exit 1
fi

# --- Test model loading, rebuild from source if architecture unsupported ---
if [ "${FORCE_BUILD}" != "1" ]; then
    echo "Testing model compatibility..."
    LOAD_OUTPUT=$(python3 -c "
import llama_cpp, sys
try:
    m = llama_cpp.Llama(model_path='${MODEL_PATH}', n_ctx=64, n_gpu_layers=0, verbose=False)
    del m
    print('OK')
except Exception as e:
    print(str(e))
" 2>/dev/null)

    if echo "${LOAD_OUTPUT}" | grep -qi "unknown model architecture"; then
        echo "Pre-built wheel does not support this model architecture."
        echo "Rebuilding llama-cpp-python from source (latest llama.cpp)..."
        install_from_source

        # Verify again after rebuild
        LOAD_OUTPUT2=$(python3 -c "
import llama_cpp
try:
    m = llama_cpp.Llama(model_path='${MODEL_PATH}', n_ctx=64, n_gpu_layers=0, verbose=False)
    del m
    print('OK')
except Exception as e:
    print(str(e))
" 2>/dev/null)

        if echo "${LOAD_OUTPUT2}" | grep -qi "unknown model architecture"; then
            echo "ERROR: Model architecture still unsupported after source build."
            echo "${LOAD_OUTPUT2}"
            exit 1
        fi
        echo "Source build supports this model."
    elif echo "${LOAD_OUTPUT}" | grep -q "OK"; then
        echo "Model compatible with installed llama-cpp-python."
    else
        # Other errors (e.g. partial bindings mismatch) are not fatal —
        # the server process may still load the model successfully.
        echo "Model test inconclusive, continuing with server launch..."
    fi
fi

echo "============================================"
echo "RunPod Deploy: llama-${RUNPOD_MODE}"
echo "============================================"
echo "Model:      ${MODEL_FILE}"
echo "Alias:      ${MODEL_ALIAS}"
echo "Context:    ${CTX_SIZE}"
echo "UBatch:     ${UBATCH_SIZE}"
echo "Port:       ${LLAMA_PORT}"
echo "GPU layers: ${N_GPU_LAYERS}"
echo "============================================"

# --- Health check function ---
health_check() {
    local max_retries=60
    local interval=15
    local count=0

    echo "Waiting for llama-server to become ready..."
    while [ $count -lt $max_retries ]; do
        if curl -sf "http://localhost:${LLAMA_PORT}/v1/models" >/dev/null 2>&1; then
            echo "llama-server is healthy!"
            return 0
        fi
        count=$((count + 1))
        echo "  Health check ${count}/${max_retries} - waiting ${interval}s..."
        sleep "${interval}"
    done

    echo "ERROR: llama-server failed to become healthy after $((max_retries * interval))s"
    return 1
}

# --- Cleanup on exit ---
cleanup() {
    echo "Shutting down llama-server (PID: ${SERVER_PID:-unknown})..."
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
    echo "Shutdown complete."
}
trap cleanup EXIT INT TERM

# --- Start llama-cpp-python server ---
echo "Starting llama-cpp-python server..."

python3 -m llama_cpp.server \
    --model "${MODEL_PATH}" \
    --model_alias "${MODEL_ALIAS}" \
    --n_gpu_layers "${N_GPU_LAYERS}" \
    --n_ctx "${CTX_SIZE}" \
    --n_batch "${UBATCH_SIZE}" \
    --flash_attn_type 1 \
    --host 0.0.0.0 \
    --port "${LLAMA_PORT}" &

SERVER_PID=$!
echo "llama-server started (PID: ${SERVER_PID})"

# --- Wait for healthy, then keep running ---
if health_check; then
    echo "============================================"
    echo "llama-${RUNPOD_MODE} is ready on port ${LLAMA_PORT}"
    echo "API endpoint: http://0.0.0.0:${LLAMA_PORT}/v1/chat/completions"
    echo "Models list:  http://0.0.0.0:${LLAMA_PORT}/v1/models"
    echo "============================================"
else
    echo "Startup failed. Check logs above."
    exit 1
fi

# Keep script alive — wait for server process
wait "${SERVER_PID}"
