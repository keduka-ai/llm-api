#!/bin/bash
# -*- coding: utf-8 -*-
# Setup Docker Script
# This script installs and configures Docker and Docker Compose

set -e  # Exit on any error

apt-get update && apt-get install -y cmake build-essential git nano
 
wget -O ai_api/models/Qwen3.5-4B-Q4_1.gguf https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_1.gguf?download=true

wget -O ai_api/models/Qwen3.5-9B-UD-Q4_K_XL.gguf https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-Q4_K_XL.gguf?download=true