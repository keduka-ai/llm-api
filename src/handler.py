"""
RunPod serverless handler for LLM inference via a local llama-server process.

The llama-server (from ggml-org/llama.cpp) is started by the entrypoint script
and exposes an OpenAI-compatible API on localhost:8080. This handler proxies
RunPod job requests to that server.
"""

import os
import re
import time
import logging
from urllib.request import Request, urlopen
from urllib.error import URLError
import json

import runpod

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
# Environment-driven configuration
# ---------------------------------------------------------------------------
RUNPOD_MODE = os.environ.get("RUNPOD_MODE", "instruct")  # "instruct" or "reasoning"

LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080")

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

# --- Health-check settings ---
_HEALTH_TIMEOUT = int(os.environ.get("LLAMA_HEALTH_TIMEOUT", 300))
_HEALTH_INTERVAL = int(os.environ.get("LLAMA_HEALTH_INTERVAL", 2))

# ---------------------------------------------------------------------------
# Cold-start: wait for llama-server to be healthy
# ---------------------------------------------------------------------------

def _wait_for_server() -> None:
    """Block until llama-server /health returns 200 or timeout expires."""
    health_url = f"{LLAMA_SERVER_URL}/health"
    deadline = time.time() + _HEALTH_TIMEOUT
    logger.info("Waiting for llama-server at %s (timeout=%ds)...", health_url, _HEALTH_TIMEOUT)

    while time.time() < deadline:
        try:
            req = Request(health_url, method="GET")
            with urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    logger.info("llama-server is healthy.")
                    return
        except (URLError, OSError):
            pass
        time.sleep(_HEALTH_INTERVAL)

    raise RuntimeError(
        f"llama-server did not become healthy within {_HEALTH_TIMEOUT}s at {health_url}"
    )


def _server_chat_completion(payload: dict) -> dict:
    """Send a chat completion request to the local llama-server."""
    url = f"{LLAMA_SERVER_URL}/v1/chat/completions"
    data = json.dumps(payload).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def _stream_chat_completion(payload: dict):
    """Stream chat completion chunks from llama-server via SSE."""
    url = f"{LLAMA_SERVER_URL}/v1/chat/completions"
    payload = {**payload, "stream": True}
    data = json.dumps(payload).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=600) as resp:
        buf = ""
        for raw_line in resp:
            buf += raw_line.decode("utf-8")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        return
                    yield json.loads(data_str)


# Wait for llama-server on cold start (skip during import-only / repo scanning)
_SKIP_HEALTH_CHECK = os.environ.get("SKIP_HEALTH_CHECK", "0") == "1"
if not _SKIP_HEALTH_CHECK:
    try:
        _wait_for_server()
    except Exception as e:
        logger.critical("llama-server not available: %s", e, exc_info=True)
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


def _streaming_generator(job_id: str, payload: dict, model_label: str, is_text_prompt: bool):
    """Yield SSE chunks from llama-server for RunPod streaming responses."""
    try:
        for chunk in _stream_chat_completion(payload):
            # Override model label in each chunk
            chunk["model"] = model_label
            if is_text_prompt:
                # Extract delta content for simplified text-prompt format
                try:
                    delta = chunk["choices"][0]["delta"]
                    content = delta.get("content", "")
                    if content:
                        yield {"response": content}
                except (KeyError, IndexError):
                    pass
            else:
                yield chunk
    except Exception as e:
        logger.error("[job=%s] Streaming error: %s", job_id, e, exc_info=True)
        yield {"error": {"message": "Streaming error occurred", "type": "server_error"}}


def handler(job: dict):
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
            messages = [
                {"role": "system", "content": system_prompt_content},
                {"role": "user", "content": prompt},
            ]

        # -----------------------------------------------------------
        # Generation parameters
        # -----------------------------------------------------------
        max_tokens = int(job_input.get("max_tokens", DEFAULT_MAX_TOKENS))
        temperature = float(job_input.get("temperature", 0.00005))
        top_p = float(job_input.get("top_p", 1.0))
        repeat_penalty = float(job_input.get("repeat_penalty", 1.2))
        think = bool(job_input.get("think", False))

        model_label = job_input.get("model") or job_input.get("model_name") or RUNPOD_MODE

        payload = dict(
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
                payload[key] = cast_fn(job_input[key])

        if "stop" in job_input:
            stop_val = job_input["stop"]
            # Accept string or list of strings
            if isinstance(stop_val, str):
                payload["stop"] = [stop_val]
            else:
                stop_list = list(stop_val)
                if len(stop_list) > MAX_STOP_SEQUENCES:
                    _log_with_job("warning", job_id, "Too many stop sequences: %d", len(stop_list))
                    return {"error": {"message": f"'stop' must not exceed {MAX_STOP_SEQUENCES} entries", "type": "invalid_request_error"}}
                payload["stop"] = stop_list

        # -----------------------------------------------------------
        # Streaming mode — return a generator for RunPod to stream
        # -----------------------------------------------------------
        stream_mode = bool(job_input.get("stream", False))

        if stream_mode:
            _log_with_job(
                "info", job_id,
                "Streaming inference (mode=%s, model=%s, max_tokens=%d, temp=%.4f, n_messages=%d)",
                RUNPOD_MODE, model_label, max_tokens, temperature, len(messages),
            )
            return _streaming_generator(job_id, payload, model_label, is_text_prompt)

        # -----------------------------------------------------------
        # Non-streaming inference via llama-server HTTP API
        # -----------------------------------------------------------
        _log_with_job(
            "info", job_id,
            "Running inference (mode=%s, model=%s, max_tokens=%d, temp=%.4f, n_messages=%d)",
            RUNPOD_MODE, model_label, max_tokens, temperature, len(messages),
        )
        start = time.time()
        result = _server_chat_completion(payload)
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

        # OpenAI-compatible response (pass through the llama-server result)
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
# Entrypoint — allow running directly or via root handler.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
