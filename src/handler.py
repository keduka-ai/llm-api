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

DEFAULT_SYSTEM_PROMPT = "You are a highly knowledgeable, kind, and helpful assistant."
DEFAULT_GENERATION_MAX_TOKENS = 75_000

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
    n_ctx = int(os.environ.get("N_CTX", cfg.get("n_ctx", DEFAULT_N_CTX.get(RUNPOD_MODE, 20_000))))
    n_ubatch = int(os.environ.get("N_UBATCH", cfg.get("n_ubatch", N_BATCH)))
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

    start = time.time()
    llm = Llama(**kwargs)
    elapsed = time.time() - start
    logger.info("Model loaded in %.2f seconds.", elapsed)
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
    try:
        job_input = job.get("input", {})
        if not job_input:
            return {"error": {"message": "Empty job input", "type": "invalid_request_error"}}

        # -----------------------------------------------------------
        # Determine input style
        # -----------------------------------------------------------
        messages = job_input.get("messages")
        prompt = job_input.get("prompt")
        is_text_prompt = messages is None and prompt is not None

        if messages is None and prompt is None:
            return {
                "error": {
                    "message": "Missing required parameter: 'messages' or 'prompt'",
                    "type": "invalid_request_error",
                }
            }

        # For text-prompt style, build a messages list
        if is_text_prompt:
            system_prompt_content = job_input.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
            full_prompt = f"{system_prompt_content}\n\n{prompt}"
            messages = [{"role": "user", "content": full_prompt}]

        # -----------------------------------------------------------
        # Generation parameters
        # -----------------------------------------------------------
        max_tokens = int(job_input.get("max_tokens", DEFAULT_GENERATION_MAX_TOKENS))
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
                create_kwargs["stop"] = list(stop_val)

        # -----------------------------------------------------------
        # Inference
        # -----------------------------------------------------------
        logger.info("Running inference (mode=%s, model_label=%s, max_tokens=%d)", RUNPOD_MODE, model_label, max_tokens)
        start = time.time()
        result = llm.create_chat_completion(**create_kwargs)
        elapsed = time.time() - start
        logger.info("Inference completed in %.2f seconds.", elapsed)

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
                logger.warning("Unexpected response structure: %s", result)
                return {"error": {"message": "Model returned no content", "type": "server_error"}}
            return {"response": response_text}

        # OpenAI-compatible response (pass through the llama-cpp result)
        # Ensure the model field reflects the requested label
        result["model"] = model_label
        return result

    except Exception as e:
        logger.error("Handler error: %s", str(e), exc_info=True)
        return {
            "error": {
                "message": f"An internal error occurred: {str(e)}",
                "type": "server_error",
            }
        }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
runpod.serverless.start({"handler": handler})
