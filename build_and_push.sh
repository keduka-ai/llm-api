#!/usr/bin/env bash
#
# Build and push the RunPod serverless Docker image to Docker Hub.
#
# Usage examples:
#   # Build with default tag (octagent-serverless), instruct mode
#   ./build_and_push.sh
#
#   # Build for reasoning mode with a custom tag
#   ./build_and_push.sh --mode reasoning --tag octagent-reasoning-v1
#
#   # Build with a model baked into the image
#   ./build_and_push.sh --model-url https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/blob/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf
#
#   # Full example
#   ./build_and_push.sh --mode instruct --tag my-image:latest --model-url https://huggingface.co/...
#

set -euo pipefail

# Defaults
MODE="instruct"
TAG="octagent-serverless"
MODEL_URL=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --model-url)
            MODEL_URL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--mode instruct|reasoning] [--tag docker-tag] [--model-url hf-url]"
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Validate mode
if [[ ! "$MODE" =~ ^(instruct|reasoning)$ ]]; then
    echo "ERROR: --mode must be 'instruct' or 'reasoning', got '${MODE}'"
    exit 1
fi

# Resolve Docker Hub username
DOCKER_USER=$(docker info 2>/dev/null | grep "Username:" | awk '{print $2}')
if [ -z "$DOCKER_USER" ]; then
    echo "Not logged in to Docker Hub. Please run 'docker login' first."
    exit 1
fi

IMAGE="${DOCKER_USER}/${TAG}"

echo "========================================="
echo "  Building RunPod Serverless Image"
echo "========================================="
echo "  Mode:      ${MODE}"
echo "  Image:     ${IMAGE}"
echo "  Model URL: ${MODEL_URL:-<none, will download at runtime>}"
echo "========================================="

# Build
docker build \
    -f Dockerfile \
    --build-arg "MODEL_URL=${MODEL_URL}" \
    -t "${IMAGE}" \
    .

echo ""
echo "Build complete: ${IMAGE}"
echo "Pushing to Docker Hub..."

docker push "${IMAGE}"

echo ""
echo "========================================="
echo "  Pushed: ${IMAGE}"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Go to https://www.runpod.io/console/serverless"
echo "  2. Create a new endpoint"
echo "  3. Set the container image to: ${IMAGE}"
echo "  4. Set environment variables:"
echo "     - RUNPOD_MODE=${MODE}"
echo "     - N_GPU_LAYERS=-1"
echo "     - FLASH_ATTN=1"
echo "  5. Configure GPU type and scaling as needed"
