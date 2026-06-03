#!/bin/bash
# Download a single GGUF model file.
#
# Usage:
#   MODEL=gemma-4-e2b-it ./download-models.sh                 # catalog alias
#   MODEL=https://hf.co/.../model.gguf ./download-models.sh   # direct HTTPS URL
#
# The downloaded filename is written to $MODELS_DIR/.active_model so that
# entrypoint.sh can auto-detect it without a separate MODEL_FILE variable.
#
# Sourceable: sourcing this file only defines resolve_model()/main() and the
# defaults — it does NOT download anything (the download is guarded below).
set -e

source "$(dirname "${BASH_SOURCE[0]}")/model-defaults.sh"

MODELS_DIR="${MODELS_DIR:-./models}"

# ---------------------------------------------------------------------------
# Model catalog — add new models here. Only HTTPS URLs are accepted (R6).
# ---------------------------------------------------------------------------
resolve_model() {
    case "$1" in
        gemma-4-e2b-it)
            MODEL_FILE="gemma-4-E2B-it-UD-Q6_K_XL.gguf"
            MODEL_URL="https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-UD-Q6_K_XL.gguf"
            ;;
        gemma-4-e2b-it-q4)
            MODEL_FILE="gemma-4-E2B-it-UD-Q4_K_XL.gguf"
            MODEL_URL="https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-UD-Q4_K_XL.gguf"
            ;;
        https://*)
            # Sanitise the filename derived from the URL: strip the query string,
            # then keep only [A-Za-z0-9._-] (R6 — no shell-meta in a baked path).
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
            echo "  gemma-4-e2b-it     gemma-4-E2B-it-UD-Q6_K_XL.gguf  (default)" >&2
            echo "  gemma-4-e2b-it-q4  gemma-4-E2B-it-UD-Q4_K_XL.gguf" >&2
            echo "" >&2
            echo "Or pass a direct HTTPS URL to any GGUF file." >&2
            exit 1
            ;;
    esac
}

main() {
    MODEL="${MODEL:-$DEFAULT_MODEL_ALIAS}"
    resolve_model "$MODEL"
    mkdir -p "$MODELS_DIR"
    echo "Downloading $MODEL_FILE from $MODEL_URL ..."
    wget -q --show-progress -O "$MODELS_DIR/$MODEL_FILE" "$MODEL_URL"
    # Marker so entrypoint.sh can auto-detect the active model.
    echo "$MODEL_FILE" > "$MODELS_DIR/.active_model"
    echo "Saved $MODEL_FILE to $MODELS_DIR/ (active model set)"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    main
fi
