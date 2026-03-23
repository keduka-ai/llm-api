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

#### Run (async)

```bash
curl -X POST https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/run \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer YOUR_API_KEY' \
    -d '{"input":{"prompt":"Your prompt"}}'
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
curl https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/status/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx \
    -H 'Authorization: Bearer YOUR_API_KEY'
```

#### RunSync (synchronous)

```bash
curl -X POST https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer YOUR_API_KEY' \
    -d '{"input":{"prompt":"Your prompt"}}'
```

#### Chat completions (OpenAI-compatible)

```bash
curl -X POST https://api.runpod.ai/v2/${RUNPOD_ENDPOINT_ID}/runsync \
    -H 'Content-Type: application/json' \
    -H 'Authorization: Bearer YOUR_API_KEY' \
    -d '{
      "input": {
        "messages": [{"role": "user", "content": "Write a Python function to sort a list."}],
        "max_tokens": 2048,
        "temperature": 0.7
      }
    }'
```

### Python examples

Create a `.env` file with your credentials:

```bash
RUNPOD_API_KEY=your_api_key_here
RUNPOD_ENDPOINT_ID=your_endpoint_id_here
```

#### Async request with polling

```python
import os
import requests
import time
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
import os
import requests
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
