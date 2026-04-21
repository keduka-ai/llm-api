#!/bin/bash
# Download a single GGUF model file.
#
# Usage:
#   MODEL=qwen3.5-9b ./download-models.sh          # use a catalog alias
#   MODEL=https://hf.co/.../model.gguf ./download-models.sh  # use a direct URL
#
# The downloaded filename is written to $MODELS_DIR/.active_model so that
# entrypoint.sh can auto-detect it without a separate MODEL_FILE variable.
set -e

source "$(dirname "$0")/model-defaults.sh"

MODELS_DIR="${MODELS_DIR:-./models}"
mkdir -p "$MODELS_DIR"

# ---------------------------------------------------------------------------
# Model catalog — add new models here
# ---------------------------------------------------------------------------
resolve_model() {
    case "$1" in
        qwen3.5-9b)
            MODEL_FILE="Qwen3.5-9B-UD-Q4_K_XL.gguf"
            MODEL_URL="https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-Q4_K_XL.gguf"
            ;;
        qwen3.5-4b)
            MODEL_FILE="Qwen3.5-4B-Q4_1.gguf"
            MODEL_URL="https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_1.gguf"
            ;;
        qwen3.6-35b)
            MODEL_FILE="Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
            MODEL_URL="https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
            ;;
        https://*)
            MODEL_FILE=$(basename "$1" | sed 's/?.*//' | tr -cd 'A-Za-z0-9._-')
            MODEL_URL="$1"
            if [ -z "$MODEL_FILE" ]; then
                echo "ERROR: could not derive filename from URL: $1" >&2
                exit 1
            fi
            ;;
        *)
            echo "ERROR: Unknown model '$1'" >&2
            echo "" >&2
            echo "Available catalog models:" >&2
            echo "  qwen3.5-9b   Qwen3.5-9B-UD-Q4_K_XL.gguf  (default)" >&2
            echo "  qwen3.5-4b   Qwen3.5-4B-Q4_1.gguf" >&2
            echo "  qwen3.6-35b  Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf" >&2
            echo "" >&2
            echo "Or pass a direct HTTPS URL to any GGUF file." >&2
            exit 1
            ;;
    esac
}

MODEL="${MODEL:-$DEFAULT_MODEL_ALIAS}"
resolve_model "$MODEL"

echo "Downloading $MODEL_FILE from $MODEL_URL ..."
wget -q --show-progress -O "$MODELS_DIR/$MODEL_FILE" "$MODEL_URL"

# Write marker so entrypoint.sh can auto-detect the model
echo "$MODEL_FILE" > "$MODELS_DIR/.active_model"
echo "Saved $MODEL_FILE to $MODELS_DIR/ (active model set)"
