# LLM API Deploy

LLM inference API supporting multiple deployment modes: Docker Compose (multi-container), RunPod pods, and RunPod serverless.

## Architecture

Two deployment paths:

1. **Docker Compose** — Django API gateway + llama-server backends + Nginx reverse proxy
2. **RunPod Serverless** — Standalone handler that loads a GGUF model and processes jobs directly (no Django/Nginx)

## RunPod Serverless Deployment

### Quick Start

```bash
# 1. Build and push the Docker image
./build_and_push.sh --mode instruct --tag octagent-serverless

# 2. Go to https://www.runpod.io/console/serverless
# 3. Create endpoint with your pushed image
# 4. Set env vars: RUNPOD_MODE=instruct, N_GPU_LAYERS=-1, FLASH_ATTN=1
```

### Bake a model into the image

```bash
./build_and_push.sh \
  --mode instruct \
  --model-url https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_1.gguf
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `RUNPOD_MODE` | `instruct` | `instruct` or `reasoning` |
| `MODELS_DIR` | `/models` | Path to GGUF model files |
| `INSTRUCT_MODEL` | `Qwen3.5-4B-Q4_1.gguf` | Instruct model filename |
| `REASONING_MODEL` | `Phi-4-mini-reasoning-UD-Q8_K_XL.gguf` | Reasoning model filename |
| `N_GPU_LAYERS` | `-1` | GPU layers (-1 = all) |
| `N_CTX` | auto | Context window size |
| `N_BATCH` | `512` | Batch size |
| `FLASH_ATTN` | `1` | Enable flash attention |
| `TENSOR_SPLIT` | `` | Multi-GPU split ratios (e.g. `0.5,0.5`) |

### Input Formats

**Chat completions (OpenAI-compatible):**
```json
{
  "input": {
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 2048,
    "temperature": 0.7
  }
}
```

**Text prompt (simple):**
```json
{
  "input": {
    "prompt": "Write a Python function to sort a list.",
    "system_prompt": "You are a helpful coding assistant.",
    "max_tokens": 2048
  }
}
```

### Files

| File | Purpose |
|---|---|
| `src/handler.py` | RunPod serverless handler |
| `Dockerfile.runpod` | Serverless Docker image |
| `requirements-runpod.txt` | Minimal serverless deps |
| `build_and_push.sh` | Build & push script |

## Docker Compose Deployment

```bash
# Start all services
docker compose up --build

# Scale API gateway
docker compose up --no-deps api nginx --build --scale api=3
```

## RunPod Pod Deployment (legacy)

```bash
# Copy project to RunPod pod
scp -i ~/.ssh/key -r ./llm-api root@<pod-ip>:/workspace

# Run on the pod
FORCE_BUILD=1 bash runpod-deploy.sh
```

## API Endpoints

```bash
# Chat completions
curl -X POST http://<host>/api/chat/completions/ \
  -H 'Content-Type: application/json' \
  -d '{"model": "instruct", "messages": [{"role": "user", "content": "Hello"}]}'

<<<<<<< HEAD

# Instruct model — basic request
curl -X POST https://194.68.245.204:8001/api/chat/completions/ \
     -H 'Content-Type: application/json' \
     -d '{
           "model": "instruct",
           "messages": [
             {"role": "user", "content": "Write a Python function to sort a list."}
           ],
           "temperature": 0.7,
           "max_tokens": 256
         }'
# llm-api
Django + DRF API gateway that routes requests to llama-cpp-python backend servers. Supports multiple model types (instruct, reasoning) via an OpenAI-compatible interface, with a Gradio chat UI (`app.py`).
=======
# Text prompt
curl -X POST http://<host>/api/text-prompt/ \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Hello", "model_name": "instruct"}'
```
>>>>>>> f23d86e (refactored for runpod)
