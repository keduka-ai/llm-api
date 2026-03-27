#!/bin/bash
# Entrypoint: start llama-server in the background, then launch the RunPod handler.
set -e

MODELS_DIR="${MODELS_DIR:-/models}"

# Resolve model filename: explicit MODEL_FILE > .active_model marker > fallback
if [ -n "$MODEL_FILE" ]; then
    MODEL_FILENAME="$MODEL_FILE"
elif [ -f "$MODELS_DIR/.active_model" ]; then
    MODEL_FILENAME=$(cat "$MODELS_DIR/.active_model")
else
    MODEL_FILENAME="Qwen3.5-9B-UD-Q4_K_XL.gguf"
fi
MODEL_PATH="${MODELS_DIR}/${MODEL_FILENAME}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: model file not found: $MODEL_PATH" >&2
    exit 1
fi

echo "Starting llama-server (model=$MODEL_FILENAME)"

# Build server args
SERVER_ARGS=(
    --model "$MODEL_PATH"
    --host 0.0.0.0
    --port 8080
    --n-gpu-layers "${N_GPU_LAYERS:--1}"
    --ctx-size "${N_CTX:-20000}"
    --batch-size "${N_BATCH:-512}"
    --ubatch-size "${N_UBATCH:-1024}"
    --jinja
    --metrics
)

# Enable reasoning format (deepseek parses <think> tags for Qwen3.5 and similar models)
REASONING_FMT="${REASONING_FORMAT:-deepseek}"
if [ "$REASONING_FMT" != "none" ]; then
    SERVER_ARGS+=(--reasoning-format "$REASONING_FMT")
fi

# --flash-attn requires a value: on, off, or auto
SERVER_ARGS+=(--flash-attn "${FLASH_ATTN_MODE:-on}")

llama-server "${SERVER_ARGS[@]}" &
LLAMA_PID=$!

python -u src/handler.py &
HANDLER_PID=$!

echo "Started llama-server (PID=$LLAMA_PID) and handler (PID=$HANDLER_PID)"

# Clean up both processes on signals (TERM/INT from Docker stop or RunPod)
cleanup() {
    echo "Shutting down..."
    kill $HANDLER_PID $LLAMA_PID 2>/dev/null
    wait $HANDLER_PID $LLAMA_PID 2>/dev/null
}
trap cleanup EXIT TERM INT

# Monitor: if either process exits, shut everything down so RunPod recycles the worker
while true; do
    if ! kill -0 $LLAMA_PID 2>/dev/null; then
        echo "ERROR: llama-server (PID=$LLAMA_PID) exited unexpectedly" >&2
        kill $HANDLER_PID 2>/dev/null
        exit 1
    fi
    if ! kill -0 $HANDLER_PID 2>/dev/null; then
        echo "ERROR: handler (PID=$HANDLER_PID) exited unexpectedly" >&2
        kill $LLAMA_PID 2>/dev/null
        exit 1
    fi
    sleep 5
done
