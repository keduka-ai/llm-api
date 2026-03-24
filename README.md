# LLM API — RunPod Serverless

RunPod serverless endpoint for LLM inference powered by the official [llama.cpp](https://github.com/ggml-org/llama.cpp) CUDA server. Supports instruct and reasoning models via an OpenAI-compatible interface, switchable at deploy time with a single environment variable.

## Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
- [Deploy on RunPod](#deploy-on-runpod)
  - [Build the Docker Image](#1-build-the-docker-image)
  - [Create a RunPod Endpoint](#2-create-a-runpod-endpoint)
- [Switching Between Instruct and Reasoning](#switching-between-instruct-and-reasoning)
- [API Usage](#api-usage)
  - [Input Styles](#input-styles)
  - [Generation Parameters](#generation-parameters)
  - [curl Examples](#curl-examples)
  - [Python Examples](#python-examples)
- [Testing](#testing)
- [Logging](#logging)
- [Project Structure](#project-structure)

---

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│  Docker container  (ghcr.io/ggml-org/llama.cpp)     │
│                                                     │
│  ┌──────────────┐   HTTP :8080   ┌───────────────┐  │
│  │ RunPod       │ ─────────────▶ │ llama-server  │  │
│  │ handler.py   │   /v1/chat/    │ (native C++)  │  │
│  │              │   completions  │               │  │
│  └──────────────┘                └───────────────┘  │
│        │                               │            │
│   RunPod API                      GPU (CUDA)        │
└─────────────────────────────────────────────────────┘
```

- **Base image**: `ghcr.io/ggml-org/llama.cpp:server-cuda` — the official llama.cpp CUDA server with up-to-date model support (Qwen3.5, Phi-4, etc.)
- **Handler** (`src/handler.py`): a RunPod serverless handler that validates requests, proxies them to the local llama-server via HTTP, and post-processes responses (think-tag stripping, format normalization)
- **Entrypoint** (`entrypoint.sh`): starts llama-server with the correct model and flags based on `RUNPOD_MODE`, then launches the handler

---

## Quick Start

```bash
git clone https://github.com/keduka-ai/llm-api.git
cd llm-api

# 1. Copy and edit environment variables
cp env.example .env
# Edit .env with your preferred settings (see Configuration section below)

# 2. Build and push the Docker image
docker login
./build_and_push.sh \
  --mode instruct \
  --tag octagent-serverless \
  --model-url https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_1.gguf

# 3. Create an endpoint on RunPod and send requests
```

---

## Configuration

### Environment Variables

Copy `env.example` to `.env` and adjust values as needed. All variables have sensible defaults and are optional.

#### Mode & Model Selection

| Variable | Default | Description |
| --- | --- | --- |
| `RUNPOD_MODE` | `instruct` | Operating mode: `instruct` or `reasoning`. Controls which model is loaded and which llama-server flags are used |
| `INSTRUCT_MODEL` | `Qwen3.5-4B-Q4_1.gguf` | Filename of the instruct model (must be in `MODELS_DIR`) |
| `REASONING_MODEL` | `Phi-4-mini-reasoning-UD-Q8_K_XL.gguf` | Filename of the reasoning model (must be in `MODELS_DIR`) |
| `MODELS_DIR` | `/models` | Directory containing GGUF model files |
| `REASONING_FORMAT` | `deepseek` | Reasoning format passed to `--reasoning-format` in reasoning mode |

#### GPU & Server Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `N_GPU_LAYERS` | `-1` | Number of layers to offload to GPU. `-1` = all layers |
| `FLASH_ATTN_MODE` | `on` | Flash attention mode: `on`, `off`, or `auto` |
| `N_CTX` | `20000` | Context window size |
| `N_BATCH` | `512` | Prompt processing batch size |
| `N_UBATCH` | `1024` | Micro-batch size |

#### Generation Defaults

| Variable | Default | Description |
| --- | --- | --- |
| `DEFAULT_MAX_TOKENS` | `4096` | Default `max_tokens` when the client doesn't specify one |
| `DEFAULT_SYSTEM_PROMPT` | `You are a highly knowledgeable, kind, and helpful assistant.` | System prompt used when the client doesn't provide one (text prompt mode only) |

#### Security Limits

| Variable | Default | Description |
| --- | --- | --- |
| `MAX_GENERATION_TOKENS` | `75000` | Hard cap on `max_tokens` to prevent resource exhaustion |
| `MAX_MESSAGES` | `256` | Maximum number of messages per request |
| `MAX_CONTENT_LENGTH` | `500000` | Maximum total characters across all messages |
| `MAX_STOP_SEQUENCES` | `16` | Maximum number of stop sequences per request |

#### Health Check

| Variable | Default | Description |
| --- | --- | --- |
| `LLAMA_SERVER_URL` | `http://127.0.0.1:8080` | URL of the llama-server (usually no need to change) |
| `LLAMA_HEALTH_TIMEOUT` | `300` | Seconds to wait for llama-server to become healthy on cold start |
| `LLAMA_HEALTH_INTERVAL` | `2` | Seconds between health check retries |

---

## Deploy on RunPod

### 1. Build the Docker Image

The `build_and_push.sh` script builds and pushes the image to Docker Hub.

```bash
docker login

# Build for instruct mode (default)
./build_and_push.sh \
  --mode instruct \
  --tag octagent-serverless \
  --model-url https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_1.gguf

# Build for reasoning mode
./build_and_push.sh --mode reasoning --tag octagent-reasoning
```

**Build script options:**

| Flag | Default | Description |
| --- | --- | --- |
| `--mode` | `instruct` | `instruct` or `reasoning` |
| `--tag` | `octagent-serverless` | Docker image tag |
| `--model-url` | _(none)_ | HTTPS URL to a GGUF model file. If omitted, `download-models.sh` runs instead |

### 2. Create a RunPod Endpoint

1. Go to [RunPod Serverless Console](https://www.runpod.io/console/serverless)
2. Click **New Endpoint**
3. Set **Container Image** to `<your-dockerhub-user>/<your-tag>` (e.g. `myuser/octagent-serverless`)
4. Set **Environment Variables** — at minimum:
   - `RUNPOD_MODE` = `instruct` (or `reasoning`)
5. _(Optional)_ Add other variables from the tables above to tune performance
6. Select your GPU type and configure scaling (min/max workers)

---

## Switching Between Instruct and Reasoning

The `RUNPOD_MODE` environment variable controls the entire behavior:

| `RUNPOD_MODE` | Model loaded | llama-server flags | Think-tag handling |
| --- | --- | --- | --- |
| `instruct` | `INSTRUCT_MODEL` (default: `Qwen3.5-4B-Q4_1.gguf`) | Standard flags, no `--reasoning-format` | Strips `<think>` tags by default |
| `reasoning` | `REASONING_MODEL` (default: `Phi-4-mini-reasoning-UD-Q8_K_XL.gguf`) | Adds `--reasoning-format deepseek` | Strips `<think>` tags by default; set `think: true` in request to preserve them |

To switch modes, change the environment variable at deploy time — no code changes or image rebuilds needed (as long as both model files are present in the image):

```bash
# Deploy as instruct
RUNPOD_MODE=instruct

# Deploy as reasoning
RUNPOD_MODE=reasoning
```

To preserve thinking content in reasoning mode responses, include `"think": true` in your request:

```json
{
  "input": {
    "messages": [{"role": "user", "content": "Solve step by step: 23 * 47"}],
    "think": true
  }
}
```

---

## API Usage

### Input Styles

The handler accepts two input styles inside the `"input"` object:

| Style | Required field | Description |
| --- | --- | --- |
| **Chat completions** | `messages` | Array of `{"role", "content"}` objects (OpenAI-compatible). Roles: `system`, `user`, `assistant` |
| **Text prompt** | `prompt` | Plain text string. Optionally include `system_prompt` (defaults to `DEFAULT_SYSTEM_PROMPT`) |

### Generation Parameters

All parameters are optional and go inside the `"input"` object:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `max_tokens` | int | `4096` | Maximum tokens to generate (capped at `MAX_GENERATION_TOKENS`) |
| `temperature` | float | `0.00005` | Sampling temperature (>= 0) |
| `top_p` | float | `1.0` | Nucleus sampling threshold (0, 1] |
| `repeat_penalty` | float | `1.2` | Repetition penalty (> 0) |
| `think` | bool | `false` | When `false`, strips `<think>...</think>` blocks from output |
| `top_k` | int | — | Top-K sampling |
| `min_p` | float | — | Min-P sampling |
| `presence_penalty` | float | — | Presence penalty |
| `frequency_penalty` | float | — | Frequency penalty |
| `seed` | int | — | Random seed for reproducibility |
| `stop` | string or list | — | Stop sequence(s) (max `MAX_STOP_SEQUENCES`) |
| `model` / `model_name` | string | — | Label echoed back in the response `model` field |

### curl Examples

#### Run (async)

```bash
curl -X POST "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run" \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -d '{"input":{"prompt":"Your prompt here"}}'
```

**Response:**

```json
{
  "id": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "status": "IN_QUEUE"
}
```

#### Check status

```bash
curl "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/status/${JOB_ID}" \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}"
```

#### RunSync (synchronous)

```bash
curl -X POST "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync" \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -d '{"input":{"prompt":"Your prompt here"}}'
```

#### Chat completions (OpenAI-compatible)

```bash
curl -X POST "https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync" \
    -H 'Content-Type: application/json' \
    -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
    -d '{
      "input": {
        "messages": [{"role": "user", "content": "Write a Python function to sort a list."}],
        "max_tokens": 2048,
        "temperature": 0.7
      }
    }'
```

### Python Examples

Set these shell variables (or create a `.env` file):

```bash
export RUNPOD_API_KEY=your_api_key_here
export RUNPOD_ENDPOINT_ID=your_endpoint_id_here
```

#### Async request with polling

```python
import os, time, requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

# Submit async job
response = requests.post(f"{BASE_URL}/run", headers=HEADERS, json={
    "input": {
        "messages": [{"role": "user", "content": "Write a Python function to sort a list."}],
        "max_tokens": 2048,
        "temperature": 0.7,
    }
})
job_id = response.json()["id"]
print(f"Job submitted: {job_id}")

# Poll for result
while True:
    status = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS).json()
    if status["status"] == "COMPLETED":
        print(status["output"])
        break
    elif status["status"] == "FAILED":
        print(f"Job failed: {status}")
        break
    time.sleep(1)
```

#### Synchronous request

```python
import os, requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]

response = requests.post(
    f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    },
    json={
        "input": {
            "prompt": "Explain recursion in simple terms.",
            "max_tokens": 2048,
        }
    },
)
print(response.json())
```

#### Using the RunPod SDK

```python
import os
from dotenv import load_dotenv
import runpod

load_dotenv()

runpod.api_key = os.environ["RUNPOD_API_KEY"]
endpoint = runpod.Endpoint(os.environ["RUNPOD_ENDPOINT_ID"])

# Async
run = endpoint.run({"input": {
    "messages": [{"role": "user", "content": "Write a Python function to sort a list."}],
    "max_tokens": 2048,
}})
output = run.output()  # blocks until complete
print(output)

# Sync
output = endpoint.run_sync({"input": {"prompt": "Explain recursion.", "max_tokens": 2048}})
print(output)
```

---

## Testing

Run the full test suite:

```bash
pip install pytest
python -m pytest tests/ -v -p no:anyio
```

The test suite covers:

| Test file | Tests | What it covers |
| --- | --- | --- |
| `tests/test_handler.py` | 106 | Handler logic: input validation, chat completions, text prompts, error handling, think-tag stripping, generation params, server communication, health checks |
| `tests/test_entrypoint.py` | 12 | Entrypoint script: mode switching, model selection, server flag construction, custom configs, error cases |

All tests run without GPU or llama-server — external dependencies are mocked.

---

## Logging

All handler logs include the RunPod job ID for traceability (`[job=<id>]`). Key events logged:

- **Cold start**: health check status, server readiness
- **Per-request**: input keys, inference parameters, mode, message count
- **Post-inference**: elapsed time, prompt/completion/total token usage
- **Errors**: client validation errors at `WARNING`, server errors at `ERROR` with full traceback

---

## Project Structure

```
├── src/
│   ├── handler.py            # RunPod serverless handler (proxies to llama-server)
│   └── __init__.py
├── tests/
│   ├── test_handler.py       # Handler test suite (106 tests)
│   └── test_entrypoint.py    # Entrypoint test suite (12 tests)
├── Dockerfile                # Based on ghcr.io/ggml-org/llama.cpp:server-cuda
├── entrypoint.sh             # Starts llama-server + handler (mode-aware)
├── build_and_push.sh         # Build & push to Docker Hub
├── download-models.sh        # Download GGUF models locally
├── requirements.txt          # Python dependencies (runpod, huggingface_hub)
└── env.example               # Example environment variables
```
