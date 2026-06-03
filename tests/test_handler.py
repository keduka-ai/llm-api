"""
Tests for src/handler.py — the RunPod serverless handler.

These run with no external services:
  * `runpod` is mocked in sys.modules BEFORE importing the handler.
  * SKIP_HEALTH_CHECK=1 stops the cold-start health gate from blocking import.
  * llama-server HTTP calls (`_server_chat_completion` / `_stream_chat_completion`)
    are patched per test.

Written as unittest.TestCase so they run under both `python -m unittest`
and `pytest` (pytest discovers/runs unittest classes natively).
"""

import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Mock `runpod` and skip the health gate BEFORE importing the handler.
# ---------------------------------------------------------------------------
_mock_runpod = types.ModuleType("runpod")
_mock_runpod.serverless = MagicMock(name="runpod.serverless")
sys.modules.setdefault("runpod", _mock_runpod)
sys.modules.setdefault("runpod.serverless", _mock_runpod.serverless)

os.environ["SKIP_HEALTH_CHECK"] = "1"
os.environ.setdefault("LLAMA_HEALTH_TIMEOUT", "1")
os.environ.setdefault("LLAMA_HEALTH_INTERVAL", "0")

# Make the repo root importable (src/, config/) regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import handler as H  # noqa: E402
from src.handler import (  # noqa: E402
    handler,
    THINK_INSTRUCTION,
    _strip_think_tags,
    _validate_messages,
    _validate_generation_params,
)


def _chat_result(content, reasoning_content=None, model="upstream-model"):
    """Build a fake llama-server OpenAI-compatible chat completion result."""
    msg = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        msg["reasoning_content"] = reasoning_content
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    }


# ---------------------------------------------------------------------------
# _strip_think_tags
# ---------------------------------------------------------------------------
class StripThinkTagsTests(unittest.TestCase):
    def test_removes_complete_block(self):
        self.assertEqual(_strip_think_tags("<think>reasoning</think>answer"), "answer")

    def test_removes_unclosed_block_to_end(self):
        self.assertEqual(_strip_think_tags("answer<think>dangling"), "answer")

    def test_keeps_text_after_orphan_close(self):
        self.assertEqual(_strip_think_tags("leftover</think>final"), "final")

    def test_no_tags_is_unchanged(self):
        self.assertEqual(_strip_think_tags("just an answer"), "just an answer")


# ---------------------------------------------------------------------------
# _validate_messages
# ---------------------------------------------------------------------------
class ValidateMessagesTests(unittest.TestCase):
    def test_not_a_list(self):
        self.assertIsNotNone(_validate_messages("nope"))

    def test_empty(self):
        self.assertIsNotNone(_validate_messages([]))

    def test_too_many(self):
        with patch.object(H, "MAX_MESSAGES", 2):
            self.assertIsNotNone(
                _validate_messages([{"role": "user", "content": "x"}] * 3)
            )

    def test_missing_role(self):
        self.assertIsNotNone(_validate_messages([{"content": "x"}]))

    def test_missing_content(self):
        self.assertIsNotNone(_validate_messages([{"role": "user"}]))

    def test_invalid_role(self):
        self.assertIsNotNone(
            _validate_messages([{"role": "root", "content": "x"}])
        )

    def test_content_too_long(self):
        with patch.object(H, "MAX_CONTENT_LENGTH", 5):
            self.assertIsNotNone(
                _validate_messages([{"role": "user", "content": "way too long"}])
            )

    def test_valid(self):
        self.assertIsNone(
            _validate_messages(
                [
                    {"role": "system", "content": "be nice"},
                    {"role": "user", "content": "hi"},
                ]
            )
        )


# ---------------------------------------------------------------------------
# _validate_generation_params
# ---------------------------------------------------------------------------
class ValidateGenerationParamsTests(unittest.TestCase):
    def test_max_tokens_non_positive(self):
        self.assertIsNotNone(_validate_generation_params({"max_tokens": 0}))

    def test_max_tokens_over_cap(self):
        with patch.object(H, "MAX_GENERATION_TOKENS", 10):
            self.assertIsNotNone(_validate_generation_params({"max_tokens": 11}))

    def test_temperature_negative(self):
        self.assertIsNotNone(_validate_generation_params({"temperature": -0.1}))

    def test_top_p_out_of_range(self):
        self.assertIsNotNone(_validate_generation_params({"top_p": 1.5}))
        self.assertIsNotNone(_validate_generation_params({"top_p": 0.0}))

    def test_repeat_penalty_non_positive(self):
        self.assertIsNotNone(_validate_generation_params({"repeat_penalty": 0}))

    def test_stop_wrong_type(self):
        self.assertIsNotNone(_validate_generation_params({"stop": 5}))

    def test_stop_list_non_strings(self):
        self.assertIsNotNone(_validate_generation_params({"stop": ["ok", 3]}))

    def test_stop_too_many(self):
        with patch.object(H, "MAX_STOP_SEQUENCES", 2):
            self.assertIsNotNone(
                _validate_generation_params({"stop": ["a", "b", "c"]})
            )

    def test_error_message_does_not_leak_value(self):
        # A bad cast must not echo the offending value back to the client.
        err = _validate_generation_params({"max_tokens": "not-an-int"})
        self.assertIsNotNone(err)
        self.assertNotIn("not-an-int", err)

    def test_valid(self):
        self.assertIsNone(
            _validate_generation_params(
                {"max_tokens": 100, "temperature": 0.2, "top_p": 0.9,
                 "repeat_penalty": 1.1, "stop": ["X"]}
            )
        )


# ---------------------------------------------------------------------------
# handler() — text-prompt input style
# ---------------------------------------------------------------------------
class HandlerTextPromptTests(unittest.TestCase):
    @patch.object(H, "_server_chat_completion")
    def test_success_returns_response_key(self, mock_chat):
        mock_chat.return_value = _chat_result("hello there")
        out = handler({"id": "j1", "input": {"prompt": "hi"}})
        self.assertEqual(out, {"response": "hello there"})

    @patch.object(H, "_server_chat_completion")
    def test_builds_system_and_user_messages(self, mock_chat):
        mock_chat.return_value = _chat_result("ok")
        handler({"id": "j", "input": {"prompt": "hi"}})
        payload = mock_chat.call_args[0][0]
        roles = [m["role"] for m in payload["messages"]]
        self.assertEqual(roles, ["system", "user"])
        self.assertEqual(payload["messages"][1]["content"], "hi")

    @patch.object(H, "_server_chat_completion")
    def test_system_prompt_override(self, mock_chat):
        mock_chat.return_value = _chat_result("ok")
        handler({"id": "j", "input": {"prompt": "hi", "system_prompt": "be terse"}})
        payload = mock_chat.call_args[0][0]
        self.assertIn("be terse", payload["messages"][0]["content"])

    @patch.object(H, "_server_chat_completion")
    def test_think_false_strips_tags(self, mock_chat):
        mock_chat.return_value = _chat_result("<think>plan</think>final")
        out = handler({"id": "j", "input": {"prompt": "hi"}})
        self.assertEqual(out, {"response": "final"})

    @patch.object(H, "_server_chat_completion")
    def test_think_true_injects_instruction(self, mock_chat):
        mock_chat.return_value = _chat_result("answer")
        handler({"id": "j", "input": {"prompt": "hi", "think": True}})
        payload = mock_chat.call_args[0][0]
        self.assertIn(THINK_INSTRUCTION, payload["messages"][0]["content"])

    @patch.object(H, "_server_chat_completion")
    def test_prompt_too_long_rejected(self, mock_chat):
        with patch.object(H, "MAX_CONTENT_LENGTH", 5):
            out = handler({"id": "j", "input": {"prompt": "way too long"}})
        self.assertIn("error", out)
        self.assertEqual(out["error"]["type"], "invalid_request_error")
        mock_chat.assert_not_called()


# ---------------------------------------------------------------------------
# handler() — chat (messages) input style
# ---------------------------------------------------------------------------
class HandlerChatTests(unittest.TestCase):
    @patch.object(H, "_server_chat_completion")
    def test_returns_openai_object_with_model_label(self, mock_chat):
        mock_chat.return_value = _chat_result("hi", model="upstream")
        out = handler(
            {"id": "j", "input": {"messages": [{"role": "user", "content": "hi"}],
                                  "model": "gemma-4-e2b-it"}}
        )
        self.assertIn("choices", out)
        self.assertEqual(out["model"], "gemma-4-e2b-it")

    @patch.object(H, "_server_chat_completion")
    def test_default_model_label_is_gemma(self, mock_chat):
        mock_chat.return_value = _chat_result("hi")
        out = handler(
            {"id": "j", "input": {"messages": [{"role": "user", "content": "hi"}]}}
        )
        self.assertEqual(out["model"], "gemma-4-e2b-it")

    @patch.object(H, "_server_chat_completion")
    def test_model_name_alias_accepted(self, mock_chat):
        mock_chat.return_value = _chat_result("hi")
        out = handler(
            {"id": "j", "input": {"messages": [{"role": "user", "content": "hi"}],
                                  "model_name": "custom-label"}}
        )
        self.assertEqual(out["model"], "custom-label")

    @patch.object(H, "_server_chat_completion")
    def test_think_false_strips_tags_and_reasoning(self, mock_chat):
        mock_chat.return_value = _chat_result(
            "<think>x</think>done", reasoning_content="x"
        )
        out = handler(
            {"id": "j", "input": {"messages": [{"role": "user", "content": "hi"}]}}
        )
        msg = out["choices"][0]["message"]
        self.assertEqual(msg["content"], "done")
        self.assertNotIn("reasoning_content", msg)

    @patch.object(H, "_server_chat_completion")
    def test_think_true_injects_into_system_message(self, mock_chat):
        mock_chat.return_value = _chat_result("answer")
        handler(
            {"id": "j", "input": {
                "messages": [{"role": "system", "content": "be nice"},
                             {"role": "user", "content": "hi"}],
                "think": True}}
        )
        payload = mock_chat.call_args[0][0]
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertIn(THINK_INSTRUCTION, payload["messages"][0]["content"])
        self.assertIn("be nice", payload["messages"][0]["content"])

    @patch.object(H, "_server_chat_completion")
    def test_invalid_messages_rejected(self, mock_chat):
        out = handler({"id": "j", "input": {"messages": []}})
        self.assertEqual(out["error"]["type"], "invalid_request_error")
        mock_chat.assert_not_called()

    @patch.object(H, "_server_chat_completion")
    def test_unknown_keys_are_ignored(self, mock_chat):
        # Forward-compat (R9): an extra/unknown field must not be rejected.
        mock_chat.return_value = _chat_result("hi")
        out = handler(
            {"id": "j", "input": {"prompt": "hi", "raw": True, "future_flag": 42}}
        )
        self.assertEqual(out, {"response": "hi"})


# ---------------------------------------------------------------------------
# handler() — generation params forwarding
# ---------------------------------------------------------------------------
class HandlerParamsTests(unittest.TestCase):
    @patch.object(H, "_server_chat_completion")
    def test_defaults_applied(self, mock_chat):
        mock_chat.return_value = _chat_result("ok")
        handler({"id": "j", "input": {"prompt": "hi"}})
        p = mock_chat.call_args[0][0]
        self.assertEqual(p["temperature"], 0.00005)
        self.assertEqual(p["top_p"], 1.0)
        self.assertEqual(p["repeat_penalty"], 1.2)
        self.assertEqual(p["max_tokens"], H.DEFAULT_MAX_TOKENS)

    @patch.object(H, "_server_chat_completion")
    def test_overrides_forwarded(self, mock_chat):
        mock_chat.return_value = _chat_result("ok")
        handler({"id": "j", "input": {
            "prompt": "hi", "max_tokens": 50, "temperature": 0.7, "top_p": 0.8,
            "repeat_penalty": 1.05}})
        p = mock_chat.call_args[0][0]
        self.assertEqual(p["max_tokens"], 50)
        self.assertEqual(p["temperature"], 0.7)
        self.assertEqual(p["top_p"], 0.8)
        self.assertEqual(p["repeat_penalty"], 1.05)

    @patch.object(H, "_server_chat_completion")
    def test_optional_params_forwarded_only_when_present(self, mock_chat):
        mock_chat.return_value = _chat_result("ok")
        handler({"id": "j", "input": {"prompt": "hi", "top_k": 40, "seed": 7}})
        p = mock_chat.call_args[0][0]
        self.assertEqual(p["top_k"], 40)
        self.assertEqual(p["seed"], 7)
        self.assertNotIn("min_p", p)

    @patch.object(H, "_server_chat_completion")
    def test_stop_string_normalised_to_list(self, mock_chat):
        mock_chat.return_value = _chat_result("ok")
        handler({"id": "j", "input": {"prompt": "hi", "stop": "STOP"}})
        self.assertEqual(mock_chat.call_args[0][0]["stop"], ["STOP"])


# ---------------------------------------------------------------------------
# handler() — error contract
# ---------------------------------------------------------------------------
class HandlerErrorTests(unittest.TestCase):
    def test_empty_input(self):
        out = handler({"id": "j", "input": {}})
        self.assertEqual(out["error"]["type"], "invalid_request_error")

    def test_missing_messages_and_prompt(self):
        out = handler({"id": "j", "input": {"temperature": 0.5}})
        self.assertEqual(out["error"]["type"], "invalid_request_error")

    @patch.object(H, "_server_chat_completion", side_effect=RuntimeError("boom"))
    def test_upstream_exception_is_server_error(self, _mock):
        out = handler({"id": "j", "input": {"prompt": "hi"}})
        self.assertEqual(out["error"]["type"], "server_error")

    @patch.object(H, "_server_chat_completion")
    def test_no_content_is_server_error(self, mock_chat):
        mock_chat.return_value = {"choices": []}
        out = handler({"id": "j", "input": {"prompt": "hi"}})
        self.assertEqual(out["error"]["type"], "server_error")

    @patch.object(H, "_server_chat_completion", side_effect=RuntimeError("secret-detail"))
    def test_error_message_does_not_leak_internals(self, _mock):
        out = handler({"id": "j", "input": {"prompt": "hi"}})
        self.assertNotIn("secret-detail", out["error"]["message"])


# ---------------------------------------------------------------------------
# handler() — streaming
# ---------------------------------------------------------------------------
class HandlerStreamingTests(unittest.TestCase):
    @patch.object(H, "_stream_chat_completion")
    def test_text_prompt_stream_yields_response_deltas(self, mock_stream):
        mock_stream.return_value = iter([
            {"choices": [{"delta": {"content": "He"}}]},
            {"choices": [{"delta": {"content": "llo"}}]},
        ])
        out = handler({"id": "j", "input": {"prompt": "hi", "stream": True}})
        chunks = list(out)
        self.assertEqual(chunks, [{"response": "He"}, {"response": "llo"}])

    @patch.object(H, "_stream_chat_completion")
    def test_chat_stream_yields_chunks_with_model(self, mock_stream):
        mock_stream.return_value = iter([
            {"choices": [{"delta": {"content": "Hi"}}], "model": "upstream"},
        ])
        out = handler({"id": "j", "input": {
            "messages": [{"role": "user", "content": "hi"}],
            "model": "gemma-4-e2b-it", "stream": True}})
        chunks = list(out)
        self.assertEqual(chunks[0]["model"], "gemma-4-e2b-it")
        self.assertIn("choices", chunks[0])


# ---------------------------------------------------------------------------
# _wait_for_server (cold-start health gate)
# ---------------------------------------------------------------------------
class WaitForServerTests(unittest.TestCase):
    def test_returns_when_healthy(self):
        resp = MagicMock()
        resp.status = 200
        resp.__enter__ = MagicMock(return_value=resp)
        resp.__exit__ = MagicMock(return_value=False)
        with patch("src.handler.urlopen", return_value=resp):
            H._wait_for_server()  # should not raise

    def test_raises_on_timeout(self):
        with patch.object(H, "_HEALTH_TIMEOUT", 0), \
             patch("src.handler.urlopen", side_effect=OSError("down")):
            with self.assertRaises(RuntimeError):
                H._wait_for_server()


if __name__ == "__main__":
    unittest.main()
