"""
Full coverage tests for src/handler.py.

The handler module calls _wait_for_server() on import and uses runpod, so we
mock those before importing. The llama-server HTTP calls are mocked via
unittest.mock.patch on _server_chat_completion.
"""

import sys
import os
import types
import json
import pytest
from unittest.mock import MagicMock, patch
from urllib.error import URLError

# ---------------------------------------------------------------------------
# Mock runpod before importing handler
# ---------------------------------------------------------------------------

_mock_runpod_module = types.ModuleType("runpod")
_mock_runpod_serverless = MagicMock(name="runpod.serverless")
_mock_runpod_module.serverless = _mock_runpod_serverless

sys.modules["runpod"] = _mock_runpod_module
sys.modules["runpod.serverless"] = _mock_runpod_serverless

os.environ["RUNPOD_MODE"] = "instruct"
os.environ["LLAMA_SERVER_URL"] = "http://127.0.0.1:8080"
os.environ["LLAMA_HEALTH_TIMEOUT"] = "1"
os.environ["LLAMA_HEALTH_INTERVAL"] = "0"

# We need to mock urlopen for the health check during import
_health_response = MagicMock()
_health_response.status = 200
_health_response.__enter__ = MagicMock(return_value=_health_response)
_health_response.__exit__ = MagicMock(return_value=False)

with patch("urllib.request.urlopen", return_value=_health_response):
    from src.handler import (
        handler,
        _strip_think_tags,
        _validate_messages,
        _validate_generation_params,
        _wait_for_server,
        _server_chat_completion,
    )


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


def _make_urlopen_response(body):
    """Create a mock context-manager response for urlopen."""
    resp = MagicMock()
    resp.status = 200
    resp.read.return_value = json.dumps(body).encode() if isinstance(body, dict) else body
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


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

    def test_nested_think_tags(self):
        # Regex is non-greedy, so inner </think> closes the first block
        result = _strip_think_tags("<think>outer<think>inner</think>middle</think>End")
        assert "End" in result

    def test_think_block_with_special_chars(self):
        assert _strip_think_tags("<think>x=1 && y<2</think>Result") == "Result"

    def test_content_between_two_blocks(self):
        assert _strip_think_tags("<think>a</think>Mid<think>b</think>End") == "MidEnd"


# ===================================================================
# _validate_messages (direct unit tests)
# ===================================================================

class TestValidateMessages:

    def test_valid_messages_returns_none(self):
        msgs = [{"role": "user", "content": "Hi"}]
        assert _validate_messages(msgs) is None

    def test_valid_multi_role_conversation(self):
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "user", "content": "Thanks"},
        ]
        assert _validate_messages(msgs) is None

    def test_not_a_list(self):
        assert "'messages' must be a list" in _validate_messages("not a list")

    def test_empty_list(self):
        assert "'messages' must not be empty" in _validate_messages([])

    def test_exceeds_max_messages(self):
        msgs = [{"role": "user", "content": "Hi"}] * 300
        result = _validate_messages(msgs)
        assert "must not exceed" in result

    def test_message_not_a_dict(self):
        result = _validate_messages(["not a dict"])
        assert "must be a dict" in result
        assert "messages[0]" in result

    def test_missing_role(self):
        result = _validate_messages([{"content": "Hi"}])
        assert "missing required field 'role'" in result

    def test_missing_content(self):
        result = _validate_messages([{"role": "user"}])
        assert "missing required field 'content'" in result

    def test_invalid_role(self):
        result = _validate_messages([{"role": "admin", "content": "Hi"}])
        assert "invalid role 'admin'" in result

    def test_total_content_length_exceeded(self):
        import src.handler as h
        original = h.MAX_CONTENT_LENGTH
        try:
            h.MAX_CONTENT_LENGTH = 10
            msgs = [{"role": "user", "content": "A" * 11}]
            result = _validate_messages(msgs)
            assert "must not exceed" in result
        finally:
            h.MAX_CONTENT_LENGTH = original

    def test_non_string_content_not_counted(self):
        """Non-string content (e.g. None) should not crash the length check."""
        msgs = [{"role": "user", "content": None}]
        assert _validate_messages(msgs) is None

    def test_error_reports_correct_index(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "bogus", "content": "Bad"},
        ]
        result = _validate_messages(msgs)
        assert "messages[1]" in result


# ===================================================================
# _validate_generation_params (direct unit tests)
# ===================================================================

class TestValidateGenerationParams:

    def test_valid_params_returns_none(self):
        assert _validate_generation_params({
            "max_tokens": 100, "temperature": 0.5, "top_p": 0.9,
            "repeat_penalty": 1.1, "stop": "END",
        }) is None

    def test_no_params_returns_none(self):
        assert _validate_generation_params({}) is None

    def test_max_tokens_zero(self):
        result = _validate_generation_params({"max_tokens": 0})
        assert "'max_tokens' must be a positive integer" in result

    def test_max_tokens_negative(self):
        result = _validate_generation_params({"max_tokens": -5})
        assert "'max_tokens' must be a positive integer" in result

    def test_max_tokens_exceeds_limit(self):
        result = _validate_generation_params({"max_tokens": 999_999_999})
        assert "must not exceed" in result

    def test_negative_temperature(self):
        result = _validate_generation_params({"temperature": -0.1})
        assert "'temperature' must be non-negative" in result

    def test_zero_temperature_valid(self):
        assert _validate_generation_params({"temperature": 0}) is None

    def test_top_p_zero(self):
        result = _validate_generation_params({"top_p": 0.0})
        assert "'top_p' must be in (0.0, 1.0]" in result

    def test_top_p_above_one(self):
        result = _validate_generation_params({"top_p": 1.1})
        assert "'top_p' must be in (0.0, 1.0]" in result

    def test_top_p_exactly_one_valid(self):
        assert _validate_generation_params({"top_p": 1.0}) is None

    def test_repeat_penalty_zero(self):
        result = _validate_generation_params({"repeat_penalty": 0})
        assert "'repeat_penalty' must be positive" in result

    def test_repeat_penalty_negative(self):
        result = _validate_generation_params({"repeat_penalty": -1.0})
        assert "'repeat_penalty' must be positive" in result

    def test_stop_as_string_valid(self):
        assert _validate_generation_params({"stop": "\n"}) is None

    def test_stop_as_list_of_strings_valid(self):
        assert _validate_generation_params({"stop": ["\n", "END"]}) is None

    def test_stop_as_integer_rejected(self):
        result = _validate_generation_params({"stop": 42})
        assert "'stop' must be a string or list of strings" in result

    def test_stop_list_with_non_strings_rejected(self):
        result = _validate_generation_params({"stop": ["ok", 123]})
        assert "'stop' list must contain only strings" in result

    def test_non_castable_max_tokens(self):
        result = _validate_generation_params({"max_tokens": "not_a_number"})
        assert "Invalid parameter type" in result

    def test_non_castable_temperature(self):
        result = _validate_generation_params({"temperature": "warm"})
        assert "Invalid parameter type" in result


# ===================================================================
# handler — chat completions
# ===================================================================

class TestHandlerChatCompletions:

    @patch("src.handler._server_chat_completion")
    def test_basic_response(self, mock_server):
        mock_server.return_value = _make_response("Hello!")
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}]}})
        assert result["choices"][0]["message"]["content"] == "Hello!"

    @patch("src.handler._server_chat_completion")
    def test_model_label_from_model_field(self, mock_server):
        mock_server.return_value = _make_response("ok")
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}], "model": "my-model"}})
        assert result["model"] == "my-model"

    @patch("src.handler._server_chat_completion")
    def test_model_label_from_model_name_field(self, mock_server):
        mock_server.return_value = _make_response("ok")
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}], "model_name": "alt"}})
        assert result["model"] == "alt"

    @patch("src.handler._server_chat_completion")
    def test_model_field_takes_priority_over_model_name(self, mock_server):
        mock_server.return_value = _make_response("ok")
        result = handler({"input": {
            "messages": [{"role": "user", "content": "Hi"}],
            "model": "primary", "model_name": "fallback",
        }})
        assert result["model"] == "primary"

    @patch("src.handler._server_chat_completion")
    def test_model_label_defaults_to_mode(self, mock_server):
        mock_server.return_value = _make_response("ok")
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}]}})
        assert result["model"] == "instruct"

    @patch("src.handler._server_chat_completion")
    def test_think_false_strips_tags(self, mock_server):
        mock_server.return_value = _make_response("<think>reasoning</think>The answer")
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}], "think": False}})
        assert result["choices"][0]["message"]["content"] == "The answer"

    @patch("src.handler._server_chat_completion")
    def test_think_false_removes_reasoning_content(self, mock_server):
        mock_server.return_value = _make_response("Answer", reasoning_content="thought")
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}], "think": False}})
        assert "reasoning_content" not in result["choices"][0]["message"]

    @patch("src.handler._server_chat_completion")
    def test_think_false_no_tags_leaves_content_alone(self, mock_server):
        """Content without think tags should pass through unchanged."""
        mock_server.return_value = _make_response("Plain answer")
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}], "think": False}})
        assert result["choices"][0]["message"]["content"] == "Plain answer"

    @patch("src.handler._server_chat_completion")
    def test_think_true_preserves_tags(self, mock_server):
        content = "<think>reasoning</think>Answer"
        mock_server.return_value = _make_response(content)
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}], "think": True}})
        assert result["choices"][0]["message"]["content"] == content

    @patch("src.handler._server_chat_completion")
    def test_think_true_preserves_reasoning_content(self, mock_server):
        mock_server.return_value = _make_response("Answer", reasoning_content="deep thought")
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}], "think": True}})
        assert result["choices"][0]["message"]["reasoning_content"] == "deep thought"

    @patch("src.handler._server_chat_completion")
    def test_think_defaults_to_false(self, mock_server):
        """When think is omitted, tags should be stripped."""
        mock_server.return_value = _make_response("<think>r</think>Answer")
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}]}})
        assert result["choices"][0]["message"]["content"] == "Answer"

    @patch("src.handler._server_chat_completion")
    def test_optional_params_forwarded(self, mock_server):
        mock_server.return_value = _make_response("ok")
        handler({"input": {
            "messages": [{"role": "user", "content": "Hi"}],
            "top_k": 50, "min_p": 0.1, "presence_penalty": 0.5,
            "frequency_penalty": 0.3, "seed": 42,
        }})
        payload = mock_server.call_args[0][0]
        assert payload["top_k"] == 50
        assert payload["min_p"] == 0.1
        assert payload["presence_penalty"] == 0.5
        assert payload["frequency_penalty"] == 0.3
        assert payload["seed"] == 42

    @patch("src.handler._server_chat_completion")
    def test_optional_params_not_sent_when_absent(self, mock_server):
        """Optional keys should not appear in the payload when not provided."""
        mock_server.return_value = _make_response("ok")
        handler({"input": {"messages": [{"role": "user", "content": "Hi"}]}})
        payload = mock_server.call_args[0][0]
        for key in ("top_k", "min_p", "presence_penalty", "frequency_penalty", "seed", "stop"):
            assert key not in payload

    @patch("src.handler._server_chat_completion")
    def test_stop_string_wrapped_in_list(self, mock_server):
        mock_server.return_value = _make_response("ok")
        handler({"input": {"messages": [{"role": "user", "content": "Hi"}], "stop": "\n"}})
        assert mock_server.call_args[0][0]["stop"] == ["\n"]

    @patch("src.handler._server_chat_completion")
    def test_stop_list_passed_through(self, mock_server):
        mock_server.return_value = _make_response("ok")
        handler({"input": {"messages": [{"role": "user", "content": "Hi"}], "stop": ["\n", "END"]}})
        assert mock_server.call_args[0][0]["stop"] == ["\n", "END"]

    @patch("src.handler._server_chat_completion")
    def test_stop_list_exceeding_max_rejected(self, mock_server):
        stops = [f"s{i}" for i in range(20)]
        result = handler({"input": {
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": stops,
        }})
        assert result["error"]["type"] == "invalid_request_error"
        assert "stop" in result["error"]["message"]
        mock_server.assert_not_called()

    @patch("src.handler._server_chat_completion")
    def test_custom_generation_params(self, mock_server):
        mock_server.return_value = _make_response("ok")
        handler({"input": {
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 100, "temperature": 0.9, "top_p": 0.95, "repeat_penalty": 1.5,
        }})
        payload = mock_server.call_args[0][0]
        assert payload["max_tokens"] == 100
        assert payload["temperature"] == 0.9
        assert payload["top_p"] == 0.95
        assert payload["repeat_penalty"] == 1.5

    @patch("src.handler._server_chat_completion")
    def test_default_generation_params(self, mock_server):
        mock_server.return_value = _make_response("ok")
        handler({"input": {"messages": [{"role": "user", "content": "Hi"}]}})
        payload = mock_server.call_args[0][0]
        assert payload["max_tokens"] == 4096
        assert payload["temperature"] == 0.00005
        assert payload["top_p"] == 1.0
        assert payload["repeat_penalty"] == 1.2

    @patch("src.handler._server_chat_completion")
    def test_messages_takes_priority_over_prompt(self, mock_server):
        """When both messages and prompt are provided, messages is used."""
        mock_server.return_value = _make_response("ok")
        result = handler({"input": {
            "messages": [{"role": "user", "content": "From messages"}],
            "prompt": "From prompt",
        }})
        # Should return OpenAI-format (not text-prompt format)
        assert "choices" in result
        assert "response" not in result
        payload = mock_server.call_args[0][0]
        assert payload["messages"][0]["content"] == "From messages"

    @patch("src.handler._server_chat_completion")
    def test_usage_info_preserved(self, mock_server):
        mock_server.return_value = _make_response("ok")
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}]}})
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 20
        assert result["usage"]["total_tokens"] == 30

    @patch("src.handler._server_chat_completion")
    def test_job_id_used(self, mock_server):
        """Handler should work with an explicit job id."""
        mock_server.return_value = _make_response("ok")
        result = handler({"id": "job-123", "input": {"messages": [{"role": "user", "content": "Hi"}]}})
        assert "choices" in result

    @patch("src.handler._server_chat_completion")
    def test_multi_choice_think_stripping(self, mock_server):
        """Think tags should be stripped from all choices, not just the first."""
        mock_server.return_value = {
            "id": "test",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "<think>a</think>First"}},
                {"index": 1, "message": {"role": "assistant", "content": "<think>b</think>Second"}},
            ],
            "usage": {},
        }
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}], "think": False}})
        assert result["choices"][0]["message"]["content"] == "First"
        assert result["choices"][1]["message"]["content"] == "Second"


# ===================================================================
# handler — text prompt
# ===================================================================

class TestHandlerTextPrompt:

    @patch("src.handler._server_chat_completion")
    def test_basic_text_prompt(self, mock_server):
        mock_server.return_value = _make_response("42")
        result = handler({"input": {"prompt": "Meaning of life?"}})
        assert result == {"response": "42"}

    @patch("src.handler._server_chat_completion")
    def test_custom_system_prompt(self, mock_server):
        mock_server.return_value = _make_response("ok")
        handler({"input": {"prompt": "Hello", "system_prompt": "You are a pirate."}})
        msgs = mock_server.call_args[0][0]["messages"]
        assert "You are a pirate." in msgs[0]["content"]
        assert "Hello" in msgs[0]["content"]

    @patch("src.handler._server_chat_completion")
    def test_default_system_prompt(self, mock_server):
        mock_server.return_value = _make_response("ok")
        handler({"input": {"prompt": "Hello"}})
        msgs = mock_server.call_args[0][0]["messages"]
        assert "highly knowledgeable" in msgs[0]["content"]

    @patch("src.handler._server_chat_completion")
    def test_strips_think_tags(self, mock_server):
        mock_server.return_value = _make_response("<think>internal</think>Clean")
        result = handler({"input": {"prompt": "Test", "think": False}})
        assert result == {"response": "Clean"}

    @patch("src.handler._server_chat_completion")
    def test_malformed_response_returns_error(self, mock_server):
        mock_server.return_value = {"choices": []}
        result = handler({"input": {"prompt": "Test"}})
        assert result["error"]["type"] == "server_error"
        assert "no content" in result["error"]["message"].lower()

    @patch("src.handler._server_chat_completion")
    def test_missing_content_key_returns_error(self, mock_server):
        mock_server.return_value = {"choices": [{"message": {}}]}
        result = handler({"input": {"prompt": "Test"}})
        assert "error" in result

    @patch("src.handler._server_chat_completion")
    def test_missing_choices_key_returns_error(self, mock_server):
        mock_server.return_value = {}
        result = handler({"input": {"prompt": "Test"}})
        assert "error" in result

    def test_prompt_not_a_string(self):
        result = handler({"input": {"prompt": 12345}})
        assert result["error"]["type"] == "invalid_request_error"
        assert "prompt" in result["error"]["message"].lower()

    def test_prompt_exceeding_max_length(self):
        import src.handler as h
        original = h.MAX_CONTENT_LENGTH
        try:
            h.MAX_CONTENT_LENGTH = 10
            result = handler({"input": {"prompt": "A" * 11}})
            assert result["error"]["type"] == "invalid_request_error"
            assert "prompt" in result["error"]["message"].lower()
        finally:
            h.MAX_CONTENT_LENGTH = original

    @patch("src.handler._server_chat_completion")
    def test_text_prompt_builds_single_user_message(self, mock_server):
        """Text prompt should produce exactly one user message."""
        mock_server.return_value = _make_response("ok")
        handler({"input": {"prompt": "Hello"}})
        msgs = mock_server.call_args[0][0]["messages"]
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"


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

    @patch("src.handler._server_chat_completion")
    def test_inference_exception(self, mock_server):
        mock_server.side_effect = RuntimeError("GPU OOM")
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}]}})
        assert result["error"]["type"] == "server_error"
        # Error message should NOT leak internal details
        assert "GPU OOM" not in result["error"]["message"]

    @patch("src.handler._server_chat_completion")
    def test_urlerror_from_server(self, mock_server):
        mock_server.side_effect = URLError("Connection refused")
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}]}})
        assert result["error"]["type"] == "server_error"
        assert "Connection refused" not in result["error"]["message"]

    @patch("src.handler._server_chat_completion")
    def test_value_error_returns_invalid_request(self, mock_server):
        """ValueError during inference should be caught as invalid_request_error."""
        mock_server.side_effect = ValueError("bad value")
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}]}})
        assert result["error"]["type"] == "invalid_request_error"

    @patch("src.handler._server_chat_completion")
    def test_type_error_returns_invalid_request(self, mock_server):
        """TypeError during inference should be caught as invalid_request_error."""
        mock_server.side_effect = TypeError("wrong type")
        result = handler({"input": {"messages": [{"role": "user", "content": "Hi"}]}})
        assert result["error"]["type"] == "invalid_request_error"

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

    def test_messages_not_a_list(self):
        result = handler({"input": {"messages": "not a list"}})
        assert result["error"]["type"] == "invalid_request_error"
        assert "must be a list" in result["error"]["message"]

    def test_empty_messages_list(self):
        result = handler({"input": {"messages": []}})
        assert result["error"]["type"] == "invalid_request_error"
        assert "must not be empty" in result["error"]["message"]

    def test_message_missing_role(self):
        result = handler({"input": {"messages": [{"content": "Hi"}]}})
        assert result["error"]["type"] == "invalid_request_error"
        assert "role" in result["error"]["message"]

    def test_message_missing_content(self):
        result = handler({"input": {"messages": [{"role": "user"}]}})
        assert result["error"]["type"] == "invalid_request_error"
        assert "content" in result["error"]["message"]

    def test_message_invalid_role(self):
        result = handler({"input": {"messages": [{"role": "tool", "content": "Hi"}]}})
        assert result["error"]["type"] == "invalid_request_error"
        assert "invalid role" in result["error"]["message"]

    def test_negative_temperature_rejected(self):
        result = handler({"input": {
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": -1,
        }})
        assert result["error"]["type"] == "invalid_request_error"
        assert "temperature" in result["error"]["message"]

    def test_top_p_zero_rejected(self):
        result = handler({"input": {
            "messages": [{"role": "user", "content": "Hi"}],
            "top_p": 0.0,
        }})
        assert result["error"]["type"] == "invalid_request_error"
        assert "top_p" in result["error"]["message"]

    def test_stop_as_integer_rejected(self):
        result = handler({"input": {
            "messages": [{"role": "user", "content": "Hi"}],
            "stop": 42,
        }})
        assert result["error"]["type"] == "invalid_request_error"
        assert "stop" in result["error"]["message"]


# ===================================================================
# _wait_for_server
# ===================================================================

class TestWaitForServer:

    @patch("src.handler.urlopen")
    def test_healthy_server_returns_immediately(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_response({"status": "ok"})
        _wait_for_server()  # Should not raise
        assert mock_urlopen.call_count == 1

    @patch("src.handler._HEALTH_TIMEOUT", 0)
    @patch("src.handler.urlopen", side_effect=OSError("Connection refused"))
    def test_timeout_on_oserror(self, _mock_urlopen):
        with pytest.raises(RuntimeError, match="did not become healthy"):
            _wait_for_server()

    @patch("src.handler._HEALTH_TIMEOUT", 0)
    @patch("src.handler.urlopen", side_effect=URLError("Name or service not known"))
    def test_timeout_on_urlerror(self, _mock_urlopen):
        with pytest.raises(RuntimeError, match="did not become healthy"):
            _wait_for_server()

    @patch("src.handler.urlopen")
    def test_retries_until_healthy(self, mock_urlopen):
        """Server returns non-200 twice, then 200 — should succeed."""
        non_200 = MagicMock()
        non_200.status = 503
        non_200.__enter__ = MagicMock(return_value=non_200)
        non_200.__exit__ = MagicMock(return_value=False)

        ok = MagicMock()
        ok.status = 200
        ok.__enter__ = MagicMock(return_value=ok)
        ok.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [non_200, non_200, ok]
        _wait_for_server()  # Should not raise
        assert mock_urlopen.call_count == 3

    @patch("src.handler.urlopen")
    def test_retries_through_connection_errors_then_succeeds(self, mock_urlopen):
        """OSError on first attempt, then healthy."""
        ok = MagicMock()
        ok.status = 200
        ok.__enter__ = MagicMock(return_value=ok)
        ok.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [OSError("refused"), ok]
        _wait_for_server()
        assert mock_urlopen.call_count == 2

    @patch("src.handler.urlopen")
    def test_health_check_url(self, mock_urlopen):
        """Verify the health check hits the correct URL."""
        mock_urlopen.return_value = _make_urlopen_response({})
        _wait_for_server()
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://127.0.0.1:8080/health"
        assert req.method == "GET"


# ===================================================================
# _server_chat_completion
# ===================================================================

class TestServerChatCompletion:

    @patch("src.handler.urlopen")
    def test_sends_json_payload(self, mock_urlopen):
        expected = _make_response("test")
        mock_urlopen.return_value = _make_urlopen_response(expected)

        payload = {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 100}
        result = _server_chat_completion(payload)

        assert result == expected
        req = mock_urlopen.call_args[0][0]
        assert "/v1/chat/completions" in req.full_url
        sent_data = json.loads(req.data)
        assert sent_data["messages"] == payload["messages"]
        assert sent_data["max_tokens"] == 100

    @patch("src.handler.urlopen")
    def test_content_type_header(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_response(_make_response("ok"))
        _server_chat_completion({"messages": []})
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Content-type") == "application/json"

    @patch("src.handler.urlopen")
    def test_uses_post_method(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_response(_make_response("ok"))
        _server_chat_completion({"messages": []})
        req = mock_urlopen.call_args[0][0]
        assert req.method == "POST"

    @patch("src.handler.urlopen")
    def test_timeout_passed_to_urlopen(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_response(_make_response("ok"))
        _server_chat_completion({"messages": []})
        _, kwargs = mock_urlopen.call_args
        assert kwargs["timeout"] == 600

    @patch("src.handler.urlopen")
    def test_url_uses_configured_server(self, mock_urlopen):
        mock_urlopen.return_value = _make_urlopen_response(_make_response("ok"))
        _server_chat_completion({"messages": []})
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "http://127.0.0.1:8080/v1/chat/completions"

    @patch("src.handler.urlopen", side_effect=URLError("Connection refused"))
    def test_urlerror_propagates(self, _mock_urlopen):
        with pytest.raises(URLError):
            _server_chat_completion({"messages": []})

    @patch("src.handler.urlopen", side_effect=TimeoutError("timed out"))
    def test_timeout_error_propagates(self, _mock_urlopen):
        with pytest.raises(TimeoutError):
            _server_chat_completion({"messages": []})
