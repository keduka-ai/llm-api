#!/bin/bash
# Entrypoint: start llama-server in the background, then launch the RunPod
# handler. If either process dies, exit non-zero so RunPod recycles the
# worker (R7.2).
#
# Sourceable: sourcing this file defines its functions/defaults without
# starting anything (the side-effecting main() is guarded at the bottom).
set -e

source "$(dirname "${BASH_SOURCE[0]}")/model-defaults.sh"

MODELS_DIR="${MODELS_DIR:-/models}"

# Resolve the model filename: explicit MODEL_FILE > .active_model marker >
# the catalog default from model-defaults.sh.
resolve_model_filename() {
    if [ -n "$MODEL_FILE" ]; then
        echo "$MODEL_FILE"
    elif [ -f "$MODELS_DIR/.active_model" ]; then
        cat "$MODELS_DIR/.active_model"
    else
        echo "$DEFAULT_MODEL_FILENAME"
    fi
}

# Assemble the llama-server argument list into the SERVER_ARGS array.
build_server_args() {
    local model_path="$1"
    SERVER_ARGS=(
        --model "$model_path"
        --alias "${MODEL_ALIAS:-$DEFAULT_MODEL_ALIAS}"
        --host 0.0.0.0
        --port "${LLAMA_PORT:-8080}"
        --n-gpu-layers "${N_GPU_LAYERS:--1}"
        --ctx-size "${N_CTX:-40192}"
        --batch-size "${N_BATCH:-4096}"
        --ubatch-size "${N_UBATCH:-2048}"
        --flash-attn "${FLASH_ATTN_MODE:-on}"
        --jinja
        --metrics
    )
    # Optional MTP draft model for speculative decoding (only if present).
    if [ -n "$DRAFT_MODEL_FILE" ] && [ -f "$MODELS_DIR/$DRAFT_MODEL_FILE" ]; then
        SERVER_ARGS+=(--model-draft "$MODELS_DIR/$DRAFT_MODEL_FILE" -ngld "${N_GPU_LAYERS_DRAFT:-99}")
    fi
    # Reasoning format is opt-in. Gemma 4 E2B-it does not emit Qwen-style
    # <think> tags by default, so this stays off unless explicitly set.
    if [ -n "$REASONING_FORMAT" ] && [ "$REASONING_FORMAT" != "none" ]; then
        SERVER_ARGS+=(--reasoning-format "$REASONING_FORMAT")
    fi
}

main() {
    local model_filename model_path
    model_filename="$(resolve_model_filename)"
    model_path="${MODELS_DIR}/${model_filename}"

    if [ ! -f "$model_path" ]; then
        echo "ERROR: model file not found: $model_path" >&2
        exit 1
    fi

    echo "Starting llama-server (model=$model_filename)"
    build_server_args "$model_path"

    llama-server "${SERVER_ARGS[@]}" &
    LLAMA_PID=$!

    python -u src/handler.py &
    HANDLER_PID=$!

    echo "Started llama-server (PID=$LLAMA_PID) and handler (PID=$HANDLER_PID)"

    # Clean up both processes on signals (TERM/INT from Docker stop / RunPod).
    cleanup() {
        echo "Shutting down..."
        kill "$HANDLER_PID" "$LLAMA_PID" 2>/dev/null
        wait "$HANDLER_PID" "$LLAMA_PID" 2>/dev/null
    }
    trap cleanup EXIT TERM INT

    # If either process exits, shut everything down so RunPod recycles the worker.
    while true; do
        if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
            echo "ERROR: llama-server (PID=$LLAMA_PID) exited unexpectedly" >&2
            kill "$HANDLER_PID" 2>/dev/null
            exit 1
        fi
        if ! kill -0 "$HANDLER_PID" 2>/dev/null; then
            echo "ERROR: handler (PID=$HANDLER_PID) exited unexpectedly" >&2
            kill "$LLAMA_PID" 2>/dev/null
            exit 1
        fi
        sleep 5
    done
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main
fi
