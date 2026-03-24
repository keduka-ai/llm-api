"""
RunPod serverless handler for LLM inference using llama-cpp-python.

Loads a GGUF model on cold start based on RUNPOD_MODE (instruct or reasoning),
then handles incoming jobs with chat-completions or text-prompt style inputs.
"""

import os
import re
import time
import logging

import runpod
from llama_cpp import Llama

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("runpod-handler")


def _log_with_job(level: str, job_id: str, msg: str, *args, **kwargs):
    """Log a message prefixed with the job ID for traceability."""
    getattr(logger, level)(f"[job={job_id}] {msg}", *args, **kwargs)

# ---------------------------------------------------------------------------
# Per-model defaults (mirrors config.MODEL_CONFIG)
# ---------------------------------------------------------------------------
MODEL_CONFIG = {
    "Qwen3.5-4B-Q4_1.gguf": {"n_ctx": 20_000, "chat_format": None, "n_ubatch": 1024},
    "Phi-4-mini-reasoning-UD-Q8_K_XL.gguf": {"n_ctx": 10_000, "chat_format": None, "n_ubatch": 1024},
    "Phi-4-mini-reasoning-Q4_K_M.gguf": {"n_ctx": 10_000, "chat_format": None, "n_ubatch": 1024},
}

DEFAULT_N_CTX = {
    "instruct": 90_000,
    "reasoning": 70_000,
}

# ---------------------------------------------------------------------------
# Environment-driven configuration
# ---------------------------------------------------------------------------
RUNPOD_MODE = os.environ.get("RUNPOD_MODE", "instruct")  # "instruct" or "reasoning"
MODELS_DIR = os.environ.get("MODELS_DIR", "/models")

INSTRUCT_MODEL = os.environ.get("INSTRUCT_MODEL", "Qwen3.5-4B-Q4_1.gguf")
REASONING_MODEL = os.environ.get("REASONING_MODEL", "Phi-4-mini-reasoning-UD-Q8_K_XL.gguf")

MODEL_FILENAMES = {
    "instruct": INSTRUCT_MODEL,
    "reasoning": REASONING_MODEL,
}

N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", -1))
N_BATCH = int(os.environ.get("N_BATCH", 512))
FLASH_ATTN = bool(int(os.environ.get("FLASH_ATTN", 1)))
USE_MMAP = bool(int(os.environ.get("USE_MMAP", 1)))
USE_MLOCK = bool(int(os.environ.get("USE_MLOCK", 1)))
MAIN_GPU = int(os.environ.get("MAIN_GPU", 0))

_tensor_split_raw = os.environ.get("TENSOR_SPLIT", "")
TENSOR_SPLIT = [float(x) for x in _tensor_split_raw.split(",") if x.strip()] or None

DEFAULT_SYSTEM_PROMPT = os.environ.get(
    "DEFAULT_SYSTEM_PROMPT",
    "You are a highly knowledgeable, kind, and helpful assistant.",
)

# --- Generation limits (all configurable via env / .env.example) ---
MAX_GENERATION_TOKENS = int(os.environ.get("MAX_GENERATION_TOKENS", 75_000))
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", 4096))
MAX_MESSAGES = int(os.environ.get("MAX_MESSAGES", 256))
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 500_000))
MAX_STOP_SEQUENCES = int(os.environ.get("MAX_STOP_SEQUENCES", 16))

# ---------------------------------------------------------------------------
# Cold-start: load model
# ---------------------------------------------------------------------------

def _load_model() -> Llama:
    """Load the GGUF model based on RUNPOD_MODE."""
    model_filename = MODEL_FILENAMES.get(RUNPOD_MODE)
    if model_filename is None:
        raise ValueError(
            f"Unknown RUNPOD_MODE '{RUNPOD_MODE}'. Expected 'instruct' or 'reasoning'."
        )

    model_path = os.path.join(MODELS_DIR, model_filename)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            f"Ensure the GGUF file is mounted at MODELS_DIR={MODELS_DIR}."
        )

    # Resolve per-model overrides from MODEL_CONFIG
    cfg = MODEL_CONFIG.get(model_filename, {})
    n_ctx = int(os.environ.get("N_CTX") or cfg.get("n_ctx") or DEFAULT_N_CTX.get(RUNPOD_MODE, 20_000))
    n_ubatch = int(os.environ.get("N_UBATCH") or cfg.get("n_ubatch") or N_BATCH)
    chat_format = cfg.get("chat_format")  # None lets llama-cpp auto-detect

    logger.info(
        "Loading model: %s (mode=%s, n_ctx=%d, n_gpu_layers=%d, flash_attn=%s)",
        model_path, RUNPOD_MODE, n_ctx, N_GPU_LAYERS, FLASH_ATTN,
    )

    kwargs = dict(
        model_path=model_path,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=n_ctx,
        n_batch=N_BATCH,
        n_ubatch=n_ubatch,
        flash_attn=FLASH_ATTN,
        use_mmap=USE_MMAP,
        use_mlock=USE_MLOCK,
        main_gpu=MAIN_GPU,
        verbose=False,
    )

    if TENSOR_SPLIT is not None:
        kwargs["tensor_split"] = TENSOR_SPLIT

    if chat_format is not None:
        kwargs["chat_format"] = chat_format

    # Enable verbose temporarily to capture backend info in logs
    kwargs["verbose"] = True

    start = time.time()
    llm = Llama(**kwargs)
    elapsed = time.time() - start
    logger.info("Model loaded in %.2f seconds.", elapsed)

    # Log model memory footprint if available
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    logger.info("Model file size: %.1f MB", model_size_mb)

    # Verify GPU offloading
    try:
        supports_gpu = Llama.supports_gpu_offload() if hasattr(Llama, "supports_gpu_offload") else None
        if supports_gpu is not None:
            logger.info("GPU offload supported by llama.cpp build: %s", supports_gpu)
            if not supports_gpu and N_GPU_LAYERS != 0:
                logger.warning(
                    "n_gpu_layers=%d requested but this llama-cpp-python build has NO GPU support! "
                    "Model is running on CPU only. Rebuild with CUDA support.",
                    N_GPU_LAYERS,
                )
        else:
            logger.info("Could not determine GPU offload support (older llama-cpp-python version)")
    except Exception as e:
        logger.warning("Could not check GPU offload support: %s", e)

    return llm


# Global model instance (loaded once on cold start)
try:
    llm = _load_model()
except Exception as e:
    logger.critical("Failed to load model on cold start: %s", e, exc_info=True)
    raise SystemExit(1)


# ---------------------------------------------------------------------------
# Think-tag stripping
# ---------------------------------------------------------------------------

_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks and any trailing partial <think> block."""
    # First remove complete <think>...</think> blocks
    cleaned = _THINK_PATTERN.sub("", text)
    # Handle unclosed <think> — remove from <think> to end
    if "<think>" in cleaned:
        cleaned = cleaned[:cleaned.index("<think>")]
    # Handle content before a closing </think> with no opening tag
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------

def _validate_messages(messages: list) -> str | None:
    """Validate the messages list format. Returns an error message or None."""
    if not isinstance(messages, list):
        return "'messages' must be a list"
    if len(messages) == 0:
        return "'messages' must not be empty"
    if len(messages) > MAX_MESSAGES:
        return f"'messages' must not exceed {MAX_MESSAGES} entries"
    total_content_length = 0
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return f"messages[{i}] must be a dict, got {type(msg).__name__}"
        if "role" not in msg:
            return f"messages[{i}] missing required field 'role'"
        if "content" not in msg:
            return f"messages[{i}] missing required field 'content'"
        if msg["role"] not in ("system", "user", "assistant"):
            return f"messages[{i}] has invalid role '{msg['role']}'"
        content = msg.get("content", "")
        if isinstance(content, str):
            total_content_length += len(content)
    if total_content_length > MAX_CONTENT_LENGTH:
        return f"Total message content must not exceed {MAX_CONTENT_LENGTH} characters"
    return None


def _validate_generation_params(job_input: dict) -> str | None:
    """Validate generation parameter types and ranges. Returns an error message or None."""
    try:
        if "max_tokens" in job_input:
            val = int(job_input["max_tokens"])
            if val <= 0:
                return "'max_tokens' must be a positive integer"
            if val > MAX_GENERATION_TOKENS:
                return f"'max_tokens' must not exceed {MAX_GENERATION_TOKENS}"
        if "temperature" in job_input:
            val = float(job_input["temperature"])
            if val < 0:
                return "'temperature' must be non-negative"
        if "top_p" in job_input:
            val = float(job_input["top_p"])
            if not (0.0 < val <= 1.0):
                return "'top_p' must be in (0.0, 1.0]"
        if "repeat_penalty" in job_input:
            val = float(job_input["repeat_penalty"])
            if val <= 0:
                return "'repeat_penalty' must be positive"
        if "stop" in job_input:
            stop_val = job_input["stop"]
            if isinstance(stop_val, str):
                pass
            elif isinstance(stop_val, list):
                if not all(isinstance(s, str) for s in stop_val):
                    return "'stop' list must contain only strings"
            else:
                return "'stop' must be a string or list of strings"
    except (ValueError, TypeError) as e:
        return f"Invalid parameter type: {e}"
    return None


def handler(job: dict) -> dict:
    """
    RunPod serverless handler.

    Accepts two input styles:

    1. Chat completions (OpenAI-compatible):
       {"messages": [{"role": "user", "content": "Hello"}], ...}

    2. Text prompt (simple):
       {"prompt": "Hello", "system_prompt": "You are helpful.", ...}

    Returns OpenAI-compatible response for chat input, or {"response": text}
    for text-prompt input.
    """
    job_id = job.get("id", "unknown")

    try:
        job_input = job.get("input", {})
        if not job_input:
            _log_with_job("warning", job_id, "Received empty job input")
            return {"error": {"message": "Empty job input", "type": "invalid_request_error"}}

        _log_with_job("info", job_id, "Job received (keys: %s)", list(job_input.keys()))

        # -----------------------------------------------------------
        # Determine input style
        # -----------------------------------------------------------
        messages = job_input.get("messages")
        prompt = job_input.get("prompt")
        is_text_prompt = messages is None and prompt is not None

        if messages is None and prompt is None:
            _log_with_job("warning", job_id, "Missing 'messages' and 'prompt'")
            return {
                "error": {
                    "message": "Missing required parameter: 'messages' or 'prompt'",
                    "type": "invalid_request_error",
                }
            }

        # -----------------------------------------------------------
        # Validate inputs
        # -----------------------------------------------------------
        if messages is not None:
            msg_error = _validate_messages(messages)
            if msg_error:
                _log_with_job("warning", job_id, "Invalid messages: %s", msg_error)
                return {"error": {"message": msg_error, "type": "invalid_request_error"}}

        param_error = _validate_generation_params(job_input)
        if param_error:
            _log_with_job("warning", job_id, "Invalid parameters: %s", param_error)
            return {"error": {"message": param_error, "type": "invalid_request_error"}}

        # For text-prompt style, build a messages list
        if is_text_prompt:
            if not isinstance(prompt, str) or len(prompt) > MAX_CONTENT_LENGTH:
                _log_with_job("warning", job_id, "Prompt too long or invalid type")
                return {"error": {"message": f"'prompt' must be a string not exceeding {MAX_CONTENT_LENGTH} characters", "type": "invalid_request_error"}}
            system_prompt_content = job_input.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
            full_prompt = f"{system_prompt_content}\n\n{prompt}"
            messages = [{"role": "user", "content": full_prompt}]

        # -----------------------------------------------------------
        # Generation parameters
        # -----------------------------------------------------------
        max_tokens = int(job_input.get("max_tokens", DEFAULT_MAX_TOKENS))
        temperature = float(job_input.get("temperature", 0.00005))
        top_p = float(job_input.get("top_p", 1.0))
        repeat_penalty = float(job_input.get("repeat_penalty", 1.2))
        think = bool(job_input.get("think", False))

        model_label = job_input.get("model") or job_input.get("model_name") or RUNPOD_MODE

        create_kwargs = dict(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
        )

        # Optional parameters — only set when explicitly provided
        _optional_keys = {
            "top_k": int,
            "min_p": float,
            "presence_penalty": float,
            "frequency_penalty": float,
            "seed": int,
        }
        for key, cast_fn in _optional_keys.items():
            if key in job_input:
                create_kwargs[key] = cast_fn(job_input[key])

        if "stop" in job_input:
            stop_val = job_input["stop"]
            # Accept string or list of strings
            if isinstance(stop_val, str):
                create_kwargs["stop"] = [stop_val]
            else:
                stop_list = list(stop_val)
                if len(stop_list) > MAX_STOP_SEQUENCES:
                    _log_with_job("warning", job_id, "Too many stop sequences: %d", len(stop_list))
                    return {"error": {"message": f"'stop' must not exceed {MAX_STOP_SEQUENCES} entries", "type": "invalid_request_error"}}
                create_kwargs["stop"] = stop_list

        # -----------------------------------------------------------
        # Inference
        # -----------------------------------------------------------
        _log_with_job(
            "info", job_id,
            "Running inference (mode=%s, model=%s, max_tokens=%d, temp=%.4f, n_messages=%d)",
            RUNPOD_MODE, model_label, max_tokens, temperature, len(messages),
        )
        start = time.time()
        result = llm.create_chat_completion(**create_kwargs)
        elapsed = time.time() - start

        # Log token usage
        usage = result.get("usage", {})
        _log_with_job(
            "info", job_id,
            "Inference completed in %.2fs (prompt_tokens=%s, completion_tokens=%s, total_tokens=%s)",
            elapsed,
            usage.get("prompt_tokens", "n/a"),
            usage.get("completion_tokens", "n/a"),
            usage.get("total_tokens", "n/a"),
        )

        # -----------------------------------------------------------
        # Post-process: strip thinking content when think=False
        # -----------------------------------------------------------
        if not think:
            for choice in result.get("choices", []):
                msg = choice.get("message", {})
                # Remove reasoning_content field if present
                msg.pop("reasoning_content", None)
                content = msg.get("content", "")
                if content and ("</think>" in content or "<think>" in content):
                    msg["content"] = _strip_think_tags(content)

        # -----------------------------------------------------------
        # Return in the appropriate format
        # -----------------------------------------------------------
        if is_text_prompt:
            # Simplified text-prompt response
            try:
                response_text = result["choices"][0]["message"]["content"]
            except (KeyError, IndexError):
                _log_with_job("warning", job_id, "Unexpected response structure: %s", result)
                return {"error": {"message": "Model returned no content", "type": "server_error"}}
            return {"response": response_text}

        # OpenAI-compatible response (pass through the llama-cpp result)
        # Ensure the model field reflects the requested label
        result["model"] = model_label
        return result

    except (ValueError, TypeError) as e:
        _log_with_job("warning", job_id, "Bad request: %s", e, exc_info=True)
        return {
            "error": {
                "message": "Invalid input: check parameter types and values",
                "type": "invalid_request_error",
            }
        }
    except Exception as e:
        _log_with_job("error", job_id, "Handler error: %s", e, exc_info=True)
        return {
            "error": {
                "message": "An internal error occurred",
                "type": "server_error",
            }
        }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
runpod.serverless.start({"handler": handler})
