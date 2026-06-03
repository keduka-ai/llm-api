#!/usr/bin/env bash
#
# Build and push the Gemma RunPod serverless image to Docker Hub.
#
# Usage:
#   # default model (gemma-4-e2b-it), default tag
#   ./build_and_push.sh
#
#   # custom tag + catalog alias
#   ./build_and_push.sh --tag gemma-e2b-serverless:v1 --model gemma-4-e2b-it
#
#   # bake a model from a direct HTTPS URL
#   ./build_and_push.sh --model-url https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-UD-Q6_K_XL.gguf
#
set -euo pipefail

TAG="gemma-e2b-serverless"
MODEL="gemma-4-e2b-it"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)        TAG="$2"; shift 2 ;;
        --model)      MODEL="$2"; shift 2 ;;
        --model-url)  MODEL="$2"; shift 2 ;;   # alias: a URL is a valid MODEL
        -h|--help)
            echo "Usage: $0 [--tag docker-tag] [--model alias | --model-url https-url]"
            exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

DOCKER_USER=$(docker info 2>/dev/null | awk '/Username:/{print $2}')
if [ -z "${DOCKER_USER}" ]; then
    echo "Not logged in to Docker Hub. Run 'docker login' first." >&2
    exit 1
fi

IMAGE="${DOCKER_USER}/${TAG}"

echo "========================================="
echo "  Building Gemma RunPod Serverless Image"
echo "========================================="
echo "  Image: ${IMAGE}"
echo "  Model: ${MODEL}"
echo "========================================="

docker build -f Dockerfile --build-arg "MODEL=${MODEL}" -t "${IMAGE}" .

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
echo "  1. https://www.runpod.io/console/serverless → New Endpoint"
echo "  2. Container image: ${IMAGE}"
echo "  3. Set a GPU type with enough VRAM for the quant + N_CTX (40192)."
echo "  4. Submit jobs to the endpoint's /run, /runsync, or /stream API."
