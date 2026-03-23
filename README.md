# LLM API — RunPod Serverless

RunPod serverless endpoint for LLM inference using llama-cpp-python with CUDA. Supports instruct and reasoning models via an OpenAI-compatible interface.

## Deploy on RunPod

### 1. Clone and build

```bash
git clone https://github.com/keduka-ai/llm-api.git
cd llm-api

# Build with a model baked into the image
docker login
./build_and_push.sh \
  --mode instruct \
  --tag octagent-serverless \
  --model-url https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_1.gguf
```

### 2. Create endpoint on RunPod

1. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Set **Container Image** to `<your-dockerhub-user>/octagent-serverless`
4. Set **Environment Variables**:
   - `RUNPOD_MODE` = `instruct`
   - `N_GPU_LAYERS` = `-1`
   - `FLASH_ATTN` = `1`
5. Select GPU type and configure scaling

### 3. Send requests

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

## Input Parameters

### Input styles

The handler accepts two input styles inside the `"input"` object:

| Style | Required field | Description |
| --- | --- | --- |
| **Chat completions** | `messages` | Array of `{"role", "content"}` objects (OpenAI-compatible). Roles: `system`, `user`, `assistant`. |
| **Text prompt** | `prompt` | Plain text string. Optionally include `system_prompt` (defaults to a general-purpose assistant prompt). |

### Generation parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `max_tokens` | int | `75000` | Maximum tokens to generate |
| `temperature` | float | `0.00005` | Sampling temperature (>= 0) |
| `top_p` | float | `1.0` | Nucleus sampling threshold (0, 1] |
| `repeat_penalty` | float | `1.2` | Repetition penalty (> 0) |
| `think` | bool | `false` | When `false`, strips `<think>...</think>` blocks from output |
| `top_k` | int | — | Top-K sampling (optional) |
| `min_p` | float | — | Min-P sampling (optional) |
| `presence_penalty` | float | — | Presence penalty (optional) |
| `frequency_penalty` | float | — | Frequency penalty (optional) |
| `seed` | int | — | Random seed for reproducibility (optional) |
| `stop` | string or list | — | Stop sequence(s) (optional) |
| `model` / `model_name` | string | — | Label echoed back in the response `model` field |

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `RUNPOD_MODE` | `instruct` | `instruct` or `reasoning` |
| `MODELS_DIR` | `/models` | Path to GGUF model files |
| `INSTRUCT_MODEL` | `Qwen3.5-4B-Q4_1.gguf` | Instruct model filename |
| `REASONING_MODEL` | `Phi-4-mini-reasoning-UD-Q8_K_XL.gguf` | Reasoning model filename |
| `N_GPU_LAYERS` | `-1` | GPU layers (-1 = all) |
| `N_CTX` | auto | Context window size (per-model defaults in `MODEL_CONFIG`) |
| `N_BATCH` | `512` | Batch size |
| `N_UBATCH` | auto | Micro-batch size (per-model defaults in `MODEL_CONFIG`) |
| `FLASH_ATTN` | `1` | Enable flash attention (0/1) |
| `USE_MMAP` | `1` | Enable memory-mapped loading (0/1) |
| `USE_MLOCK` | `1` | Lock model in RAM (0/1) |
| `MAIN_GPU` | `0` | Primary GPU index |
| `TENSOR_SPLIT` | — | Multi-GPU split ratios (e.g. `0.5,0.5`) |

## Logging

All handler logs include the RunPod job ID for traceability (`[job=<id>]`). Key events logged:

- **Cold start**: model path, mode, context size, GPU config, load time, file size
- **Per-request**: input keys, inference parameters, mode, message count
- **Post-inference**: elapsed time, prompt/completion/total token usage
- **Errors**: client validation errors at `WARNING`, server errors at `ERROR` with full traceback

## Project Structure

```
├── src/
│   ├── handler.py         # RunPod serverless handler
│   └── __init__.py
├── Dockerfile             # Serverless Docker image (CUDA 12.4)
├── requirements.txt       # Python dependencies (runpod, huggingface_hub)
├── build_and_push.sh      # Build & push to Docker Hub
└── download-models.sh     # Download GGUF models locally
```
