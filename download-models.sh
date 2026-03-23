#!/bin/bash
# Download GGUF model files
set -e

MODELS_DIR="${MODELS_DIR:-./models}"
mkdir -p "$MODELS_DIR"

wget -O "$MODELS_DIR/Qwen3.5-4B-Q4_1.gguf" \
  "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_1.gguf?download=true"

# wget -O "$MODELS_DIR/Qwen3.5-9B-UD-Q4_K_XL.gguf" \
#   "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-Q4_K_XL.gguf?download=true"
