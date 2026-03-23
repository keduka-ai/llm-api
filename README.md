# LLM API Deploy

Django + DRF API gateway that routes requests to llama-cpp-python backend servers. Supports multiple model types (instruct, reasoning) via an OpenAI-compatible interface, with a Gradio chat UI (`app.py`).

## Deploy on RunPod (Serverless)

### 1. Clone the repo on your local machine

```bash
git clone https://github.com/keduka-ai/llm-api.git
cd llm-api
```

### 2. Build and push the Docker image

```bash
# Log in to Docker Hub
docker login

# Build for instruct mode with model baked in
./build_and_push.sh \
  --mode instruct \
  --tag octagent-serverless \
  --model-url https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_1.gguf

# Or for reasoning mode
./build_and_push.sh \
  --mode reasoning \
  --tag octagent-reasoning
```

### 3. Create the serverless endpoint on RunPod

1. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Set **Container Image** to `<your-dockerhub-user>/octagent-serverless`
4. Set **Environment Variables**:
   - `RUNPOD_MODE` = `instruct` (or `reasoning`)
   - `N_GPU_LAYERS` = `-1`
   - `FLASH_ATTN` = `1`
5. Select a GPU type (e.g. A40, A100, L40S) and configure scaling

### 4. Send requests

Use the RunPod endpoint URL from the console:

```bash
# Chat completions (OpenAI-compatible)
curl -X POST https://api.runpod.ai/v2/<endpoint-id>/runsync \
  -H 'Authorization: Bearer <RUNPOD_API_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": {
      "messages": [{"role": "user", "content": "Write a Python function to sort a list."}],
      "max_tokens": 2048,
      "temperature": 0.7
    }
  }'

# Text prompt (simple format)
curl -X POST https://api.runpod.ai/v2/<endpoint-id>/runsync \
  -H 'Authorization: Bearer <RUNPOD_API_KEY>' \
  -H 'Content-Type: application/json' \
  -d '{
    "input": {
      "prompt": "Explain recursion in simple terms.",
      "system_prompt": "You are a helpful coding assistant.",
      "max_tokens": 2048
    }
  }'
```

## Deploy on RunPod (Pod — legacy)

For running on a dedicated RunPod GPU pod instead of serverless:

```bash
# Clone on the pod
git clone https://github.com/keduka-ai/llm-api.git
cd llm-api

# Download models
bash download-models.sh

# Start the llama-server
RUNPOD_MODE=instruct FORCE_BUILD=1 bash runpod-deploy.sh
```

## Deploy with Docker Compose (self-hosted)

```bash
git clone https://github.com/keduka-ai/llm-api.git
cd llm-api

# Place GGUF models in ai_api/models/
bash download-models.sh

# Copy and configure environment
cp env.example.tmp .env
# Edit .env with your settings

# Start all services
docker compose up --build

# Scale API gateway
docker compose up --no-deps api nginx --build --scale api=3
```

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `RUNPOD_MODE` | `instruct` | `instruct` or `reasoning` |
| `MODELS_DIR` | `/models` | Path to GGUF model files |
| `INSTRUCT_MODEL` | `Qwen3.5-4B-Q4_1.gguf` | Instruct model filename |
| `REASONING_MODEL` | `Phi-4-mini-reasoning-UD-Q8_K_XL.gguf` | Reasoning model filename |
| `N_GPU_LAYERS` | `-1` | GPU layers (-1 = all) |
| `N_CTX` | auto | Context window size |
| `N_BATCH` | `512` | Batch size |
| `FLASH_ATTN` | `1` | Enable flash attention |
| `TENSOR_SPLIT` | | Multi-GPU split ratios (e.g. `0.5,0.5`) |

## Project Structure

| Path | Purpose |
| --- | --- |
| `src/handler.py` | RunPod serverless handler |
| `Dockerfile.runpod` | Serverless Docker image |
| `build_and_push.sh` | Build & push to Docker Hub |
| `requirements-runpod.txt` | Minimal serverless deps |
| `ai_api/` | Django API app (views, models, backends) |
| `project/` | Django project settings |
| `Dockerfile` | Docker Compose API gateway image |
| `docker-compose.yaml` | Multi-container deployment |
| `runpod-deploy.sh` | RunPod pod deployment script |
| `app.py` | Gradio chat UI |
