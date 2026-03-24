"""
Full coverage tests for src/handler.py.

The handler module runs cold-start code (model loading + runpod.serverless.start)
on import, so we mock llama_cpp.Llama and runpod before importing it.
"""

import sys
import os
import types
import pytest
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock heavy dependencies before importing handler
# ---------------------------------------------------------------------------

_mock_llama_module = types.ModuleType("llama_cpp")
_mock_llama_cls = MagicMock(name="Llama")
_mock_llama_module.Llama = _mock_llama_cls  # type: ignore[attr-defined]
_mock_llama_module.llama_supports_gpu_offload = MagicMock(return_value=True)  # type: ignore[attr-defined]

_mock_runpod_module = types.ModuleType("runpod")
_mock_runpod_serverless = MagicMock(name="runpod.serverless")
_mock_runpod_module.serverless = _mock_runpod_serverless

sys.modules["llama_cpp"] = _mock_llama_module
sys.modules["runpod"] = _mock_runpod_module
sys.modules["runpod.serverless"] = _mock_runpod_serverless

# Create fake model file for cold-start loading
_TEST_MODELS_DIR = os.path.join(os.path.dirname(__file__), "_test_models")
os.makedirs(_TEST_MODELS_DIR, exist_ok=True)
_FAKE_MODEL = os.path.join(_TEST_MODELS_DIR, "Qwen3.5-4B-Q4_1.gguf")
with open(_FAKE_MODEL, "wb") as f:
    f.write(b"\x00" * 1024)

os.environ["RUNPOD_MODE"] = "instruct"
os.environ["MODELS_DIR"] = _TEST_MODELS_DIR
os.environ["INSTRUCT_MODEL"] = "Qwen3.5-4B-Q4_1.gguf"
os.environ["N_GPU_LAYERS"] = "0"
os.environ["FLASH_ATTN"] = "0"
os.environ["USE_MMAP"] = "0"
os.environ["USE_MLOCK"] = "0"

# Import handler (cold start runs with mocked Llama)
from src.handler import handler, _strip_think_tags, _load_model  # noqa: E402

_mock_llm_instance = _mock_llama_cls.return_value


# ---------------------------------------------------------------------------
# Helper: fake chat completion response
# ---------------------------------------------------------------------------

def _make_response(content, reasoning_content=None):
    msg = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        msg["reasoning_content"] = reasoning_content
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }


# ===================================================================
# _strip_think_tags
# ===================================================================

class TestStripThinkTags:

    def test_complete_block_removed(self):
        assert _strip_think_tags("<think>reasoning</think>Hello") == "Hello"

    def test_multiple_blocks(self):
        assert _strip_think_tags("<think>a</think>Hello <think>b</think>World") == "Hello World"

    def test_multiline_block(self):
        assert _strip_think_tags("<think>\nline1\nline2\n</think>Answer") == "Answer"

    def test_unclosed_think_tag(self):
        assert _strip_think_tags("Start<think>this goes on forever") == "Start"

    def test_orphaned_closing_tag(self):
        assert _strip_think_tags("hidden</think>Visible") == "Visible"

    def test_no_think_tags(self):
        assert _strip_think_tags("Just plain text") == "Just plain text"

    def test_empty_string(self):
        assert _strip_think_tags("") == ""

    def test_only_think_tags(self):
        assert _strip_think_tags("<think>everything</think>") == ""

    def test_empty_think_block(self):
        assert _strip_think_tags("<think></think>Hello") == "Hello"

    def test_whitespace_around_block(self):
        assert _strip_think_tags("  <think>stuff</think>  Answer  ") == "Answer"


# ===================================================================
# handler — chat completions
# ===================================================================

class TestHandlerChatCompletions:

    def test_basic_response(self):
        _mock_llm_instance.create_chat_completion.return_value = _make_response("Hello!")
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}]}})
        assert result["choices"][0]["message"]["content"] == "Hello!"

    def test_model_label_from_model_field(self):
        _mock_llm_instance.create_chat_completion.return_value = _make_response("ok")
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}], "model": "my-model"}})
        assert result["model"] == "my-model"

    def test_model_label_from_model_name_field(self):
        _mock_llm_instance.create_chat_completion.return_value = _make_response("ok")
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}], "model_name": "alt"}})
        assert result["model"] == "alt"

    def test_model_label_defaults_to_mode(self):
        _mock_llm_instance.create_chat_completion.return_value = _make_response("ok")
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}]}})
        assert result["model"] == "instruct"

    def test_think_false_strips_tags(self):
        _mock_llm_instance.create_chat_completion.return_value = _make_response(
            "<think>reasoning</think>The answer"
        )
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}], "think": False}})
        assert result["choices"][0]["message"]["content"] == "The answer"

    def test_think_false_removes_reasoning_content(self):
        _mock_llm_instance.create_chat_completion.return_value = _make_response(
            "Answer", reasoning_content="thought"
        )
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}], "think": False}})
        assert "reasoning_content" not in result["choices"][0]["message"]

    def test_think_true_preserves_tags(self):
        content = "<think>reasoning</think>Answer"
        _mock_llm_instance.create_chat_completion.return_value = _make_response(content)
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}], "think": True}})
        assert result["choices"][0]["message"]["content"] == content

    def test_think_true_preserves_reasoning_content(self):
        _mock_llm_instance.create_chat_completion.return_value = _make_response(
            "Answer", reasoning_content="deep thought"
        )
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}], "think": True}})
        assert result["choices"][0]["message"]["reasoning_content"] == "deep thought"

    def test_optional_params_forwarded(self):
        _mock_llm_instance.create_chat_completion.return_value = _make_response("ok")
        handler({"input": {
            "messages": [{"role": "user", "content": "Hi"}],
            "top_k": 50, "min_p": 0.1, "presence_penalty": 0.5,
            "frequency_penalty": 0.3, "seed": 42,
        }})
        kw = _mock_llm_instance.create_chat_completion.call_args[1]
        assert kw["top_k"] == 50
        assert kw["min_p"] == 0.1
        assert kw["presence_penalty"] == 0.5
        assert kw["frequency_penalty"] == 0.3
        assert kw["seed"] == 42

    def test_stop_string_wrapped_in_list(self):
        _mock_llm_instance.create_chat_completion.return_value = _make_response("ok")
        handler({"input": {"messages": [{"role": "user", "content": "Hi"}], "stop": "\n"}})
        assert _mock_llm_instance.create_chat_completion.call_args[1]["stop"] == ["\n"]

    def test_stop_list_passed_through(self):
        _mock_llm_instance.create_chat_completion.return_value = _make_response("ok")
        handler({"input": {"messages": [{"role": "user", "content": "Hi"}], "stop": ["\n", "END"]}})
        assert _mock_llm_instance.create_chat_completion.call_args[1]["stop"] == ["\n", "END"]

    def test_custom_generation_params(self):
        _mock_llm_instance.create_chat_completion.return_value = _make_response("ok")
        handler({"input": {
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100, "temperature": 0.9, "top_p": 0.95, "repeat_penalty": 1.5,
        }})
        kw = _mock_llm_instance.create_chat_completion.call_args[1]
        assert kw["max_tokens"] == 100
        assert kw["temperature"] == 0.9
        assert kw["top_p"] == 0.95
        assert kw["repeat_penalty"] == 1.5

    def test_default_generation_params(self):
        _mock_llm_instance.create_chat_completion.return_value = _make_response("ok")
        handler({"input": {"messages": [{"role": "user", "content": "Hi"}]}})
        kw = _mock_llm_instance.create_chat_completion.call_args[1]
        assert kw["max_tokens"] == 4096
        assert kw["temperature"] == 0.00005
        assert kw["top_p"] == 1.0
        assert kw["repeat_penalty"] == 1.2


# ===================================================================
# handler — text prompt
# ===================================================================

class TestHandlerTextPrompt:

    def test_basic_text_prompt(self):
        _mock_llm_instance.create_chat_completion.return_value = _make_response("42")
        result = handler({"input": {"prompt": "Meaning of life?"}})
        assert result == {"response": "42"}

    def test_custom_system_prompt(self):
        _mock_llm_instance.create_chat_completion.return_value = _make_response("ok")
        handler({"input": {"prompt": "Hello", "system_prompt": "You are a pirate."}})
        msgs = _mock_llm_instance.create_chat_completion.call_args[1]["messages"]
        assert "You are a pirate." in msgs[0]["content"]
        assert "Hello" in msgs[0]["content"]

    def test_default_system_prompt(self):
        _mock_llm_instance.create_chat_completion.return_value = _make_response("ok")
        handler({"input": {"prompt": "Hello"}})
        msgs = _mock_llm_instance.create_chat_completion.call_args[1]["messages"]
        assert "highly knowledgeable" in msgs[0]["content"]

    def test_strips_think_tags(self):
        _mock_llm_instance.create_chat_completion.return_value = _make_response(
            "<think>internal</think>Clean"
        )
        result = handler({"input": {"prompt": "Test", "think": False}})
        assert result == {"response": "Clean"}

    def test_malformed_response_returns_error(self):
        _mock_llm_instance.create_chat_completion.return_value = {"choices": []}
        result = handler({"input": {"prompt": "Test"}})
        assert "error" in result
        assert result["error"]["type"] == "server_error"

    def test_missing_content_key_returns_error(self):
        _mock_llm_instance.create_chat_completion.return_value = {
            "choices": [{"message": {}}]
        }
        result = handler({"input": {"prompt": "Test"}})
        assert "error" in result


# ===================================================================
# handler — error cases
# ===================================================================

class TestHandlerErrors:

    def test_empty_input(self):
        result = handler({"input": {}})
        assert result["error"]["type"] == "invalid_request_error"

    def test_missing_input_key(self):
        result = handler({})
        assert "Empty job input" in result["error"]["message"]

    def test_missing_messages_and_prompt(self):
        result = handler({"input": {"temperature": 0.5}})
        assert "Missing required parameter" in result["error"]["message"]

    def test_inference_exception(self):
        _mock_llm_instance.create_chat_completion.side_effect = RuntimeError("GPU OOM")
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}]}})
        assert result["error"]["type"] == "server_error"
        # Error message should NOT leak internal details
        assert "GPU OOM" not in result["error"]["message"]
        _mock_llm_instance.create_chat_completion.side_effect = None

    def test_max_tokens_exceeds_limit(self):
        result = handler({"input": {
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 999_999_999,
        }})
        assert result["error"]["type"] == "invalid_request_error"
        assert "exceed" in result["error"]["message"]

    def test_stop_non_string_items_rejected(self):
        result = handler({"input": {
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": [123, None],
        }})
        assert result["error"]["type"] == "invalid_request_error"
        assert "stop" in result["error"]["message"]

    def test_too_many_messages_rejected(self):
        msgs = [{"role": "user", "content": "Hi"}] * 300
        result = handler({"input": {"messages": msgs}})
        assert result["error"]["type"] == "invalid_request_error"
        assert "exceed" in result["error"]["message"].lower()


# ===================================================================
# _load_model
# ===================================================================

class TestLoadModel:

    def test_unknown_mode_raises(self):
        import src.handler as h
        original = h.RUNPOD_MODE
        try:
            h.RUNPOD_MODE = "nonexistent"
            with pytest.raises(ValueError, match="Unknown RUNPOD_MODE"):
                _load_model()
        finally:
            h.RUNPOD_MODE = original

    def test_missing_model_file_raises(self):
        import src.handler as h
        original = h.MODELS_DIR
        try:
            h.MODELS_DIR = "/nonexistent/path"
            with pytest.raises(FileNotFoundError, match="Model file not found"):
                _load_model()
        finally:
            h.MODELS_DIR = original

    def test_successful_load_kwargs(self):
        _mock_llama_cls.reset_mock()
        _load_model()
        _mock_llama_cls.assert_called_once()
        kw = _mock_llama_cls.call_args[1]
        assert kw["model_path"] == _FAKE_MODEL
        assert kw["n_gpu_layers"] == 0
        assert kw["flash_attn"] is False
        assert kw["verbose"] is True

    def test_gpu_check_raises_when_no_gpu_support(self):
        """When GPU layers requested but build lacks GPU support, raise RuntimeError."""
        from unittest.mock import patch
        import src.handler as h
        original = h.N_GPU_LAYERS
        try:
            h.N_GPU_LAYERS = -1
            _mock_llama_cls.reset_mock()
            with patch("src.handler.llama_supports_gpu_offload", return_value=False):
                with pytest.raises(RuntimeError, match="NO GPU support"):
                    _load_model()
        finally:
            h.N_GPU_LAYERS = original

    def test_gpu_check_passes_when_layers_zero(self):
        """When n_gpu_layers=0, no GPU check failure even without GPU support."""
        from unittest.mock import patch
        _mock_llama_cls.reset_mock()
        with patch("src.handler.llama_supports_gpu_offload", return_value=False):
            _load_model()  # N_GPU_LAYERS=0 from env setup, should not raise

    def test_gpu_check_exception_logged_as_warning(self):
        """When llama_supports_gpu_offload raises an unexpected error, log warning and continue."""
        from unittest.mock import patch
        _mock_llama_cls.reset_mock()
        with patch("src.handler.llama_supports_gpu_offload", side_effect=OSError("libcuda not found")):
            llm = _load_model()  # Should not raise
            assert llm is not None


# ===================================================================
# Cleanup
# ===================================================================

@pytest.fixture(autouse=True, scope="session")
def cleanup_test_models():
    yield
    import shutil
    shutil.rmtree(_TEST_MODELS_DIR, ignore_errors=True)
