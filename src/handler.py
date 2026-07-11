"""
RunPod serverless handler for Gemma inference via a local llama-server process.

`entrypoint.sh` starts llama-server (ggml-org/llama.cpp) in the background; it
exposes an OpenAI-compatible API on localhost:8080. This handler proxies RunPod
job requests to that server.

The job I/O contract is a **drop-in** with the reference RunPod LLM service
(`llm-api-deploy`): same accepted input fields, same response/error/streaming
shapes (see SHARED_RULES R9). The one model-specific difference is reasoning
handling — Gemma 4 E2B-it does not use Qwen's `/think` / `/no_think` directives,
so `think=True` injects a natural-language THINK_INSTRUCTION into the system
message (matching the existing Gemma wrapper) and `think=False` strips any
`<think>...</think>` blocks from the output.
"""

import os
import re
import time
import json
import logging
from urllib.request import Request, urlopen
from urllib.error import URLError

import runpod

# ---------------------------------------------------------------------------
# Logging — never log full prompts or completions (R6); counts/timing only.
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("runpod-handler")


def _log_with_job(level, job_id, msg, *args, **kwargs):
    getattr(logger, level)(f"[job={job_id}] {msg}", *args, **kwargs)


# ---------------------------------------------------------------------------
# Environment-driven configuration
# ---------------------------------------------------------------------------
LLAMA_SERVER_URL = os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080")

# Default system prompt: rather than a static assistant persona, instruct the
# model to draft a task-specific expert persona inside <persona>...</persona>
# tags and a short plan inside <plan>...</plan> tags before answering. Both
# blocks are stripped from non-streaming output the same way <think> blocks
# are, so callers only ever see the final answer.
PERSONA_INSTRUCTION = (
    "First, silently determine the subject, topic, and sub-topic of the "
    "user's request, and write a concise persona for the expert best suited "
    "to answer it inside <persona>...</persona> tags — \"You are an expert "
    "in ... Your main objective is ...\" — in under 150 words. Then, acting "
    "as that persona, answer the user's request. Never mention the persona "
    "or the tags in your final answer."
)
PLAN_INSTRUCTION = (
    "Next, before answering, write a short numbered plan of the steps your "
    "answer will follow inside <plan>...</plan> tags — no more than 5 steps. "
    "Then follow that plan in your answer. Never mention the plan or the "
    "tags in your final answer."
)
DEFAULT_SYSTEM_PROMPT = os.environ.get(
    "DEFAULT_SYSTEM_PROMPT", f"{PERSONA_INSTRUCTION}\n\n{PLAN_INSTRUCTION}"
)
DEFAULT_MODEL_LABEL = os.environ.get("DEFAULT_MODEL_NAME", "gemma-4-e2b-it")

# Generation limits / defaults (all env-configurable).
MAX_GENERATION_TOKENS = int(os.environ.get("MAX_GENERATION_TOKENS", 40_192))
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", 4096))
MAX_MESSAGES = int(os.environ.get("MAX_MESSAGES", 256))
MAX_CONTENT_LENGTH = int(os.environ.get("MAX_CONTENT_LENGTH", 500_000))
MAX_STOP_SEQUENCES = int(os.environ.get("MAX_STOP_SEQUENCES", 16))

# Cold-start health gate.
_HEALTH_TIMEOUT = int(os.environ.get("LLAMA_HEALTH_TIMEOUT", 300))
_HEALTH_INTERVAL = int(os.environ.get("LLAMA_HEALTH_INTERVAL", 2))

# Default generation params (kept aligned with the reference service, R9.3).
_DEFAULT_TEMPERATURE = 0.00005
_DEFAULT_TOP_P = 1.0
_DEFAULT_REPEAT_PENALTY = 1.2

# Optional generation params forwarded to llama-server only when provided.
_OPTIONAL_PARAM_CASTS = {
    "top_k": int,
    "min_p": float,
    "presence_penalty": float,
    "frequency_penalty": float,
    "seed": int,
}

# Reasoning directive injected when think=True (Gemma-specific, R9.4).
THINK_INSTRUCTION = (
    "Before answering, reason through the problem step by step inside "
    "<think>...</think> tags. After the closing </think> tag, give the "
    "final answer to the user and nothing else."
)

_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
_PERSONA_PATTERN = re.compile(r"<persona>.*?</persona>", re.DOTALL)
_PLAN_PATTERN = re.compile(r"<plan>.*?</plan>", re.DOTALL)


# ---------------------------------------------------------------------------
# Cold-start: wait for llama-server to be healthy
# ---------------------------------------------------------------------------
def _wait_for_server():
    """Block until llama-server /health returns 200 or timeout expires."""
    health_url = f"{LLAMA_SERVER_URL}/health"
    deadline = time.time() + _HEALTH_TIMEOUT
    logger.info("Waiting for llama-server at %s (timeout=%ds)...", health_url, _HEALTH_TIMEOUT)
    while time.time() < deadline:
        try:
            with urlopen(Request(health_url, method="GET"), timeout=5) as resp:
                if resp.status == 200:
                    logger.info("llama-server is healthy.")
                    return
        except (URLError, OSError):
            pass
        time.sleep(_HEALTH_INTERVAL)
    raise RuntimeError(
        f"llama-server did not become healthy within {_HEALTH_TIMEOUT}s at {health_url}"
    )


def _server_chat_completion(payload):
    """POST a chat completion to the local llama-server and return the JSON."""
    url = f"{LLAMA_SERVER_URL}/v1/chat/completions"
    data = json.dumps(payload).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def _stream_chat_completion(payload):
    """Yield chat-completion chunks from llama-server via SSE."""
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


# Wait for llama-server on cold start (skipped during import-only / tests).
if os.environ.get("SKIP_HEALTH_CHECK", "0") != "1":
    try:
        _wait_for_server()
    except Exception as e:  # pragma: no cover - exercised in the container
        logger.critical("llama-server not available: %s", e, exc_info=True)
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _strip_tag_blocks(text, pattern, open_tag, close_tag):
    """Remove <tag>...</tag> blocks plus any dangling partial block."""
    cleaned = pattern.sub("", text)
    if open_tag in cleaned:
        cleaned = cleaned[: cleaned.index(open_tag)]
    if close_tag in cleaned:
        cleaned = cleaned.split(close_tag)[-1]
    return cleaned.strip()


def _strip_think_tags(text):
    """Remove <think>...</think> blocks and any trailing partial block."""
    return _strip_tag_blocks(text, _THINK_PATTERN, "<think>", "</think>")


def _strip_persona_tags(text):
    """Remove <persona>...</persona> blocks and any trailing partial block."""
    return _strip_tag_blocks(text, _PERSONA_PATTERN, "<persona>", "</persona>")


def _strip_plan_tags(text):
    """Remove <plan>...</plan> blocks and any trailing partial block."""
    return _strip_tag_blocks(text, _PLAN_PATTERN, "<plan>", "</plan>")


def _error(message, error_type):
    """Uniform error envelope (R9.2)."""
    return {"error": {"message": message, "type": error_type}}


def _validate_messages(messages):
    """Validate the messages list. Returns an error string or None."""
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


def _validate_generation_params(job_input):
    """Validate generation param types/ranges. Returns an error string or None."""
    try:
        if "max_tokens" in job_input:
            val = int(job_input["max_tokens"])
            if val <= 0:
                return "'max_tokens' must be a positive integer"
            if val > MAX_GENERATION_TOKENS:
                return f"'max_tokens' must not exceed {MAX_GENERATION_TOKENS}"
        if "temperature" in job_input:
            if float(job_input["temperature"]) < 0:
                return "'temperature' must be non-negative"
        if "top_p" in job_input:
            if not (0.0 < float(job_input["top_p"]) <= 1.0):
                return "'top_p' must be in (0.0, 1.0]"
        if "repeat_penalty" in job_input:
            if float(job_input["repeat_penalty"]) <= 0:
                return "'repeat_penalty' must be positive"
        if "stop" in job_input:
            stop_val = job_input["stop"]
            if isinstance(stop_val, str):
                pass
            elif isinstance(stop_val, list):
                if not all(isinstance(s, str) for s in stop_val):
                    return "'stop' list must contain only strings"
                if len(stop_val) > MAX_STOP_SEQUENCES:
                    return f"'stop' must not exceed {MAX_STOP_SEQUENCES} entries"
            else:
                return "'stop' must be a string or list of strings"
    except (ValueError, TypeError) as e:
        # Surface only the exception class, never str(e) — the offending value
        # may contain user data we must not echo back (R6).
        return f"Invalid parameter type: {type(e).__name__}"
    return None


def _apply_think(messages, think):
    """Inject THINK_INSTRUCTION into the system message when think=True."""
    if not think:
        return messages
    messages = list(messages)
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        existing = messages[0].get("content", "")
        merged = (
            f"{THINK_INSTRUCTION}\n\n{existing}"
            if isinstance(existing, str) and existing
            else THINK_INSTRUCTION
        )
        messages[0] = {"role": "system", "content": merged}
    else:
        messages.insert(0, {"role": "system", "content": THINK_INSTRUCTION})
    return messages


def _strip_reasoning(result):
    """Remove reasoning_content + <think> blocks from each choice message."""
    for choice in result.get("choices", []):
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message")
        if not isinstance(msg, dict):
            continue
        msg.pop("reasoning_content", None)
        content = msg.get("content", "")
        if content and ("<think>" in content or "</think>" in content):
            msg["content"] = _strip_think_tags(content)


def _strip_meta(result):
    """Remove <persona> and <plan> blocks from each choice message.

    Unlike think stripping this is unconditional: these blocks are
    handler-injected meta-content (see PERSONA_INSTRUCTION and
    PLAN_INSTRUCTION), never something the caller asked to see.
    """
    for choice in result.get("choices", []):
        if not isinstance(choice, dict):
            continue
        msg = choice.get("message")
        if not isinstance(msg, dict):
            continue
        content = msg.get("content", "")
        if content and ("<persona>" in content or "</persona>" in content):
            content = _strip_persona_tags(content)
        if content and ("<plan>" in content or "</plan>" in content):
            content = _strip_plan_tags(content)
        msg["content"] = content


class _StreamTagFilter:
    """Incrementally strip <persona>/<plan> blocks from streamed text.

    Streaming deltas can split a tag anywhere (`<perso` + `na>`), so plain
    per-chunk regex stripping cannot work. This filter emits text as soon as
    it provably cannot belong to one of the tags, holds back anything that
    could still turn into a tag, and discards everything inside an open
    block. `flush()` releases a held partial prefix that never became a tag;
    a block left open at stream end stays suppressed (matching the
    non-streaming dangling-open behavior).
    """

    TAGS = ("persona", "plan")

    def __init__(self, tags=None):
        self._tags = tuple(tags) if tags is not None else self.TAGS
        self._buf = ""
        self._in_block = None  # tag name while inside a block, else None

    def feed(self, text):
        """Add streamed text; return the part that is safe to emit."""
        self._buf += text
        out = []
        while self._buf:
            if self._in_block:
                close = f"</{self._in_block}>"
                idx = self._buf.find(close)
                if idx == -1:
                    # Discard block text, keeping only a tail that could be
                    # the start of the close tag split across chunks.
                    keep = self._partial_suffix_len(self._buf, close)
                    self._buf = self._buf[len(self._buf) - keep:] if keep else ""
                    break
                self._buf = self._buf[idx + len(close):]
                self._in_block = None
                continue
            lt = self._buf.find("<")
            if lt == -1:
                out.append(self._buf)
                self._buf = ""
                break
            out.append(self._buf[:lt])
            self._buf = self._buf[lt:]
            opened = False
            possible_prefix = False
            for tag in self._tags:
                open_tag = f"<{tag}>"
                if self._buf.startswith(open_tag):
                    self._buf = self._buf[len(open_tag):]
                    self._in_block = tag
                    opened = True
                    break
                if open_tag.startswith(self._buf):
                    possible_prefix = True
            if opened:
                continue
            if possible_prefix:
                break  # hold until more text disambiguates
            out.append("<")
            self._buf = self._buf[1:]
        return "".join(out)

    def flush(self):
        """Release held text at stream end (unless inside an open block)."""
        tail = "" if self._in_block else self._buf
        self._buf = ""
        self._in_block = None
        return tail

    @staticmethod
    def _partial_suffix_len(buf, token):
        """Longest k < len(token) with buf ending in token's first k chars."""
        for k in range(min(len(buf), len(token) - 1), 0, -1):
            if buf.endswith(token[:k]):
                return k
        return 0


def _streaming_generator(job_id, payload, model_label, is_text_prompt, think):
    """Yield SSE chunks from llama-server for RunPod streaming responses.

    <persona>/<plan> blocks are stripped from the streamed text via
    _StreamTagFilter, mirroring the unconditional _strip_meta() pass on
    non-streaming responses. When think=False, <think> blocks and delta
    reasoning_content are stripped too, mirroring _strip_reasoning().
    """
    tags = _StreamTagFilter.TAGS if think else _StreamTagFilter.TAGS + ("think",)
    tag_filter = _StreamTagFilter(tags)
    try:
        for chunk in _stream_chat_completion(payload):
            try:
                delta = chunk["choices"][0]["delta"]
                content = delta.get("content", "")
            except (KeyError, IndexError, AttributeError, TypeError):
                delta, content = None, ""
            if delta is not None and not think:
                delta.pop("reasoning_content", None)
            if is_text_prompt:
                emit = tag_filter.feed(content) if content else ""
                if emit:
                    yield {"response": emit}
            else:
                if delta is not None and content:
                    delta["content"] = tag_filter.feed(content)
                chunk["model"] = model_label
                yield chunk
        tail = tag_filter.flush()
        if tail:
            if is_text_prompt:
                yield {"response": tail}
            else:
                yield {
                    "choices": [{"index": 0, "delta": {"content": tail}}],
                    "model": model_label,
                    "object": "chat.completion.chunk",
                }
    except Exception as e:
        _log_with_job("error", job_id, "Streaming error: %s", e, exc_info=True)
        yield {"error": {"message": "Streaming error occurred", "type": "server_error"}}


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------
def handler(job):
    """
    RunPod serverless handler.

    Input styles (under job["input"]):
      1. Chat:        {"messages": [...], ...}  → OpenAI-compatible response
      2. Text prompt: {"prompt": "...", ...}    → {"response": text}
    """
    job_id = job.get("id", "unknown")
    try:
        job_input = job.get("input", {})
        if not job_input:
            _log_with_job("warning", job_id, "Empty job input")
            return _error("Empty job input", "invalid_request_error")

        messages = job_input.get("messages")
        prompt = job_input.get("prompt")
        is_text_prompt = messages is None and prompt is not None

        if messages is None and prompt is None:
            return _error(
                "Missing required parameter: 'messages' or 'prompt'",
                "invalid_request_error",
            )

        if messages is not None:
            msg_error = _validate_messages(messages)
            if msg_error:
                _log_with_job("warning", job_id, "Invalid messages: %s", msg_error)
                return _error(msg_error, "invalid_request_error")

        param_error = _validate_generation_params(job_input)
        if param_error:
            _log_with_job("warning", job_id, "Invalid parameters: %s", param_error)
            return _error(param_error, "invalid_request_error")

        if is_text_prompt:
            if not isinstance(prompt, str) or len(prompt) > MAX_CONTENT_LENGTH:
                return _error(
                    f"'prompt' must be a string not exceeding {MAX_CONTENT_LENGTH} characters",
                    "invalid_request_error",
                )
            system_content = job_input.get("system_prompt") or DEFAULT_SYSTEM_PROMPT
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]

        think = bool(job_input.get("think", False))
        max_tokens = int(job_input.get("max_tokens", DEFAULT_MAX_TOKENS))
        temperature = float(job_input.get("temperature", _DEFAULT_TEMPERATURE))
        top_p = float(job_input.get("top_p", _DEFAULT_TOP_P))
        repeat_penalty = float(job_input.get("repeat_penalty", _DEFAULT_REPEAT_PENALTY))
        model_label = job_input.get("model") or job_input.get("model_name") or DEFAULT_MODEL_LABEL

        messages = _apply_think(messages, think)

        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
        }
        for key, cast_fn in _OPTIONAL_PARAM_CASTS.items():
            if key in job_input:
                payload[key] = cast_fn(job_input[key])
        if "stop" in job_input:
            stop_val = job_input["stop"]
            payload["stop"] = [stop_val] if isinstance(stop_val, str) else list(stop_val)

        if bool(job_input.get("stream", False)):
            _log_with_job(
                "info", job_id,
                "Streaming (think=%s, model=%s, max_tokens=%d, n_messages=%d)",
                think, model_label, max_tokens, len(messages),
            )
            return _streaming_generator(
                job_id, payload, model_label, is_text_prompt, think
            )

        _log_with_job(
            "info", job_id,
            "Inference (think=%s, model=%s, max_tokens=%d, n_messages=%d)",
            think, model_label, max_tokens, len(messages),
        )
        start = time.time()
        result = _server_chat_completion(payload)
        elapsed = time.time() - start

        if not isinstance(result, dict):
            return _error("upstream returned unexpected shape", "server_error")

        usage = result.get("usage", {}) if isinstance(result.get("usage"), dict) else {}
        _log_with_job(
            "info", job_id,
            "Completed in %.2fs (prompt_tokens=%s, completion_tokens=%s)",
            elapsed, usage.get("prompt_tokens", "n/a"), usage.get("completion_tokens", "n/a"),
        )

        if not think:
            _strip_reasoning(result)
        _strip_meta(result)

        if is_text_prompt:
            choices = result.get("choices")
            if not isinstance(choices, list) or not choices:
                return _error("Model returned no content", "server_error")
            first = choices[0] if isinstance(choices[0], dict) else {}
            message = first.get("message") if isinstance(first.get("message"), dict) else {}
            response_text = message.get("content")
            if not isinstance(response_text, str):
                return _error("Model returned no content", "server_error")
            return {"response": response_text}

        result["model"] = model_label
        return result

    except (ValueError, TypeError) as e:
        _log_with_job("warning", job_id, "Bad request: %s", e)
        return _error(
            "Invalid input: check parameter types and values",
            "invalid_request_error",
        )
    except Exception as e:
        _log_with_job("error", job_id, "Handler error: %s", e, exc_info=True)
        return _error("An internal error occurred", "server_error")


# ---------------------------------------------------------------------------
# Allow running directly or via the root handler.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
