"""
Tests for client/llm_api_client — the client-side CustomLLM / Agent wrappers
for the RunPod serverless worker.

These run with no external services: `urlopen` is patched inside the client
module so every test asserts on the exact Request the client would send and
feeds back a canned response body.

Written as unittest.TestCase so they run under both `python -m unittest`
and `pytest` (pytest discovers/runs unittest classes natively).
"""

import asyncio
import io
import json
import os
import sys
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

# Make the client package importable regardless of cwd (mirrors what a real
# consumer gets from `pip install llm-api-client`).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "client"))

import llm_api_client as llms  # noqa: E402
from llm_api_client import Agent, CustomLLM  # noqa: E402


class _FakeResponse(io.BytesIO):
    """Minimal context-manager response like the one urlopen returns."""

    def __init__(self, body, status=200):
        super().__init__(json.dumps(body).encode())
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _patched_call(body, **llm_kwargs):
    """Call a CustomLLM with urlopen patched; return (result, sent Request)."""
    llm = CustomLLM(system_message="You are a test assistant.", **llm_kwargs)
    with patch.object(llms, "urlopen", return_value=_FakeResponse(body)) as mock_open:
        result = llm("hello")
    request = mock_open.call_args[0][0]
    return result, request


def _sent_input(request):
    """Decode the flat job-input dict out of a captured Request."""
    payload = json.loads(request.data.decode())
    return payload["input"]


class TestRequestConstruction(unittest.TestCase):
    def test_payload_is_wrapped_in_runpod_input_envelope(self):
        _, request = _patched_call({"output": {"response": "hi"}})
        payload = json.loads(request.data.decode())
        self.assertIn("input", payload)
        self.assertEqual(payload["input"]["prompt"], "hello")

    def test_default_fields_match_handler_contract(self):
        _, request = _patched_call({"output": {"response": "hi"}})
        sent = _sent_input(request)
        self.assertEqual(sent["system_prompt"], "You are a test assistant.")
        self.assertEqual(sent["model_name"], llms.DEFAULT_MODEL_NAME)
        self.assertEqual(sent["temperature"], llms.DEFAULT_TEMPERATURE)
        self.assertEqual(sent["repeat_penalty"], llms.DEFAULT_REPEAT_PENALTY)
        self.assertEqual(sent["max_tokens"], llms.DEFAULT_MAX_TOKENS)
        self.assertEqual(sent["seed"], 42)
        self.assertFalse(sent["think"])

    def test_content_type_header_is_json(self):
        _, request = _patched_call({"output": {"response": "hi"}})
        self.assertEqual(request.get_header("Content-type"), "application/json")

    def test_api_key_env_sets_bearer_header(self):
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "rp-test-key"}):
            _, request = _patched_call({"output": {"response": "hi"}})
        self.assertEqual(request.get_header("Authorization"), "Bearer rp-test-key")

    def test_no_auth_header_without_api_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RUNPOD_API_KEY", None)
            _, request = _patched_call({"output": {"response": "hi"}})
        self.assertFalse(request.has_header("Authorization"))

    def test_per_call_overrides_take_precedence(self):
        llm = CustomLLM(system_message="base", temperature=0.5, max_tokens=128)
        with patch.object(
            llms, "urlopen", return_value=_FakeResponse({"output": {"response": "hi"}})
        ) as mock_open:
            llm(
                "hello",
                system_message="override",
                temperature=0.9,
                max_tokens=64,
                model="other-model",
                think=True,
                seed=7,
            )
        sent = _sent_input(mock_open.call_args[0][0])
        self.assertEqual(sent["system_prompt"], "override")
        self.assertEqual(sent["temperature"], 0.9)
        self.assertEqual(sent["max_tokens"], 64)
        self.assertEqual(sent["model_name"], "other-model")
        self.assertTrue(sent["think"])
        self.assertEqual(sent["seed"], 7)

    def test_custom_payload_merges_but_prompt_wins(self):
        llm = CustomLLM(system_message="base")
        with patch.object(
            llms, "urlopen", return_value=_FakeResponse({"output": {"response": "hi"}})
        ) as mock_open:
            llm("hello", custom_payload={"top_p": 0.8, "prompt": "clobbered"})
        sent = _sent_input(mock_open.call_args[0][0])
        self.assertEqual(sent["top_p"], 0.8)
        self.assertEqual(sent["prompt"], "hello")

    def test_url_defaults_to_module_default(self):
        llm = CustomLLM(system_message="base")
        self.assertEqual(llm.url, llms.DEFAULT_LLM_URL)

    def test_explicit_url_is_used(self):
        llm = CustomLLM(system_message="base", url="http://example.test/runsync")
        with patch.object(
            llms, "urlopen", return_value=_FakeResponse({"output": {"response": "hi"}})
        ) as mock_open:
            llm("hello")
        self.assertEqual(
            mock_open.call_args[0][0].full_url, "http://example.test/runsync"
        )


class TestResponseParsing(unittest.TestCase):
    def test_runsync_output_response(self):
        result, _ = _patched_call({"output": {"response": "hi there"}})
        self.assertEqual(result, "hi there")

    def test_flat_response_without_envelope(self):
        result, _ = _patched_call({"response": "flat hi"})
        self.assertEqual(result, "flat hi")

    def test_openai_choices_shape(self):
        body = {
            "output": {
                "choices": [{"message": {"role": "assistant", "content": "chatty"}}]
            }
        }
        result, _ = _patched_call(body)
        self.assertEqual(result, "chatty")

    def test_handler_error_envelope_raises(self):
        body = {"output": {"error": {"message": "boom", "type": "server_error"}}}
        with self.assertRaises(RuntimeError):
            _patched_call(body)

    def test_failed_job_status_raises(self):
        body = {"status": "FAILED", "error": "worker exploded"}
        with self.assertRaises(RuntimeError):
            _patched_call(body)

    def test_empty_completion_returns_empty_string(self):
        result, _ = _patched_call({"output": {"response": ""}})
        self.assertEqual(result, "")

    def test_non_json_body_raises(self):
        llm = CustomLLM(system_message="base")
        fake = _FakeResponse({"unused": True})
        fake.seek(0)
        fake.truncate()
        fake.write(b"<html>gateway timeout</html>")
        fake.seek(0)
        with patch.object(llms, "urlopen", return_value=fake):
            with self.assertRaises(RuntimeError):
                llm("hello")


class TestAgent(unittest.TestCase):
    def test_default_system_message_contains_agent_name(self):
        agent = Agent(agent_name="Kedu Ka")
        self.assertIn("Kedu Ka", agent.system_message)

    def test_explicit_system_message_wins(self):
        agent = Agent(system_message="custom sys")
        self.assertEqual(agent.system_message, "custom sys")

    def test_call_sends_agent_configuration(self):
        agent = Agent(system_message="agent sys", max_tokens=99, temperature=0.3)
        with patch.object(
            llms, "urlopen", return_value=_FakeResponse({"output": {"response": "ok"}})
        ) as mock_open:
            result = agent("do the thing")
        sent = _sent_input(mock_open.call_args[0][0])
        self.assertEqual(result, "ok")
        self.assertEqual(sent["system_prompt"], "agent sys")
        self.assertEqual(sent["max_tokens"], 99)
        self.assertEqual(sent["temperature"], 0.3)

    def test_call_overrides_agent_configuration(self):
        agent = Agent(system_message="agent sys")
        with patch.object(
            llms, "urlopen", return_value=_FakeResponse({"output": {"response": "ok"}})
        ) as mock_open:
            agent("prompt", system_message="call sys", think=True)
        sent = _sent_input(mock_open.call_args[0][0])
        self.assertEqual(sent["system_prompt"], "call sys")
        self.assertTrue(sent["think"])

    def test_async_call_matches_sync(self):
        agent = Agent(system_message="agent sys")
        with patch.object(
            llms, "urlopen", return_value=_FakeResponse({"output": {"response": "ok"}})
        ):
            result = asyncio.run(agent.async_call("prompt"))
        self.assertEqual(result, "ok")

    def test_str_returns_agent_name(self):
        self.assertEqual(str(Agent(agent_name="Namey")), "Namey")


def _http_error(code, body, url="http://localhost:8000/runsync"):
    """Build an HTTPError whose body can be read (like a real urlopen error)."""
    payload = body if isinstance(body, (bytes, bytearray)) else json.dumps(body).encode()
    return HTTPError(url, code, "error", None, io.BytesIO(payload))


_OK_BODY = {"output": {"response": "hi"}}


class TestHttpErrorSurfacing(unittest.TestCase):
    def test_error_body_message_is_surfaced(self):
        llm = CustomLLM(system_message="s")
        error = _http_error(
            400,
            {"error": {"message": "'max_tokens' must be a positive integer",
                       "type": "invalid_request_error"}},
        )
        with patch.object(llms, "urlopen", side_effect=error):
            with self.assertRaises(RuntimeError) as ctx:
                llm("hello")
        self.assertIn("max_tokens", str(ctx.exception))
        self.assertIn("400", str(ctx.exception))

    def test_client_errors_are_not_retried(self):
        llm = CustomLLM(system_message="s")
        with patch.object(llms, "_sleep") as mock_sleep:
            with patch.object(
                llms, "urlopen", side_effect=_http_error(400, {"error": "bad"})
            ) as mock_open:
                with self.assertRaises(RuntimeError):
                    llm("hello")
        self.assertEqual(mock_open.call_count, 1)
        mock_sleep.assert_not_called()

    def test_non_json_error_body_still_raises_with_status(self):
        llm = CustomLLM(system_message="s", max_retries=0)
        with patch.object(
            llms, "urlopen", side_effect=_http_error(502, b"<html>bad gateway</html>")
        ):
            with self.assertRaises(RuntimeError) as ctx:
                llm("hello")
        self.assertIn("502", str(ctx.exception))


class TestRetries(unittest.TestCase):
    def test_transient_http_error_is_retried_then_succeeds(self):
        llm = CustomLLM(system_message="s")
        responses = [_http_error(503, {"error": "overloaded"}), _FakeResponse(_OK_BODY)]
        with patch.object(llms, "_sleep") as mock_sleep:
            with patch.object(llms, "urlopen", side_effect=responses) as mock_open:
                result = llm("hello")
        self.assertEqual(result, "hi")
        self.assertEqual(mock_open.call_count, 2)
        mock_sleep.assert_called_once()

    def test_transport_error_is_retried_then_succeeds(self):
        llm = CustomLLM(system_message="s")
        responses = [URLError("connection refused"), _FakeResponse(_OK_BODY)]
        with patch.object(llms, "_sleep"):
            with patch.object(llms, "urlopen", side_effect=responses) as mock_open:
                result = llm("hello")
        self.assertEqual(result, "hi")
        self.assertEqual(mock_open.call_count, 2)

    def test_gives_up_after_max_retries(self):
        llm = CustomLLM(system_message="s", max_retries=1)
        responses = [
            _http_error(503, {"error": "overloaded"}),
            _http_error(503, {"error": {"message": "still overloaded"}}),
        ]
        with patch.object(llms, "_sleep"):
            with patch.object(llms, "urlopen", side_effect=responses) as mock_open:
                with self.assertRaises(RuntimeError) as ctx:
                    llm("hello")
        self.assertEqual(mock_open.call_count, 2)
        self.assertIn("still overloaded", str(ctx.exception))

    def test_transport_error_exhausted_propagates(self):
        llm = CustomLLM(system_message="s", max_retries=0)
        with patch.object(llms, "urlopen", side_effect=URLError("refused")):
            with self.assertRaises(URLError):
                llm("hello")


class TestQueuedJobs(unittest.TestCase):
    def test_polls_status_until_completed(self):
        llm = CustomLLM(system_message="s")
        responses = [
            _FakeResponse({"id": "job-1", "status": "IN_QUEUE"}),
            _FakeResponse({"id": "job-1", "status": "IN_PROGRESS"}),
            _FakeResponse({"status": "COMPLETED", "output": {"response": "late answer"}}),
        ]
        with patch.object(llms, "_sleep"):
            with patch.object(llms, "urlopen", side_effect=responses) as mock_open:
                result = llm("hello")
        self.assertEqual(result, "late answer")
        self.assertEqual(mock_open.call_count, 3)
        status_request = mock_open.call_args_list[1][0][0]
        self.assertEqual(status_request.full_url, "http://localhost:8000/status/job-1")
        self.assertEqual(status_request.get_method(), "GET")

    def test_queued_response_without_job_id_raises(self):
        llm = CustomLLM(system_message="s")
        with patch.object(
            llms, "urlopen", return_value=_FakeResponse({"status": "IN_QUEUE"})
        ):
            with self.assertRaises(RuntimeError):
                llm("hello")

    def test_poll_timeout_raises(self):
        llm = CustomLLM(system_message="s", timeout=0)
        responses = [
            _FakeResponse({"id": "job-1", "status": "IN_QUEUE"}),
            _FakeResponse({"id": "job-1", "status": "IN_PROGRESS"}),
        ]
        with patch.object(llms, "_sleep"):
            with patch.object(llms, "urlopen", side_effect=responses):
                with self.assertRaises(RuntimeError) as ctx:
                    llm("hello")
        self.assertIn("job-1", str(ctx.exception))

    def test_polled_job_failure_raises(self):
        llm = CustomLLM(system_message="s")
        responses = [
            _FakeResponse({"id": "job-1", "status": "IN_QUEUE"}),
            _FakeResponse({"status": "FAILED", "error": "worker exploded"}),
        ]
        with patch.object(llms, "_sleep"):
            with patch.object(llms, "urlopen", side_effect=responses):
                with self.assertRaises(RuntimeError):
                    llm("hello")


class TestStreaming(unittest.TestCase):
    def test_stream_yields_chunks_in_order(self):
        llm = CustomLLM(system_message="s")
        responses = [
            _FakeResponse({"id": "job-9"}),
            _FakeResponse({
                "status": "IN_PROGRESS",
                "stream": [{"output": {"response": "Hel"}},
                           {"output": {"response": "lo"}}],
            }),
            _FakeResponse({"status": "COMPLETED",
                           "stream": [{"output": {"response": "!"}}]}),
        ]
        with patch.object(llms, "_sleep"):
            with patch.object(llms, "urlopen", side_effect=responses) as mock_open:
                chunks = list(llm.stream("hello"))
        self.assertEqual(chunks, ["Hel", "lo", "!"])
        submit = mock_open.call_args_list[0][0][0]
        self.assertEqual(submit.full_url, "http://localhost:8000/run")
        self.assertTrue(json.loads(submit.data.decode())["input"]["stream"])
        poll = mock_open.call_args_list[1][0][0]
        self.assertEqual(poll.full_url, "http://localhost:8000/stream/job-9")
        self.assertEqual(poll.get_method(), "GET")

    def test_stream_error_chunk_raises(self):
        llm = CustomLLM(system_message="s")
        responses = [
            _FakeResponse({"id": "job-9"}),
            _FakeResponse({
                "status": "IN_PROGRESS",
                "stream": [{"output": {"error": {"message": "boom",
                                                 "type": "server_error"}}}],
            }),
        ]
        with patch.object(llms, "_sleep"):
            with patch.object(llms, "urlopen", side_effect=responses):
                with self.assertRaises(RuntimeError) as ctx:
                    list(llm.stream("hello"))
        self.assertIn("boom", str(ctx.exception))

    def test_stream_failed_status_raises(self):
        llm = CustomLLM(system_message="s")
        responses = [
            _FakeResponse({"id": "job-9"}),
            _FakeResponse({"status": "FAILED", "stream": [], "error": "dead"}),
        ]
        with patch.object(llms, "_sleep"):
            with patch.object(llms, "urlopen", side_effect=responses):
                with self.assertRaises(RuntimeError):
                    list(llm.stream("hello"))

    def test_stream_without_job_id_raises(self):
        llm = CustomLLM(system_message="s")
        with patch.object(llms, "urlopen", return_value=_FakeResponse({})):
            with self.assertRaises(RuntimeError):
                list(llm.stream("hello"))


class TestLazyConfig(unittest.TestCase):
    def test_llm_url_env_is_read_at_construction_time(self):
        with patch.dict(os.environ, {"LLM_URL": "http://env.example/runsync"}):
            llm = CustomLLM(system_message="s")
        self.assertEqual(llm.url, "http://env.example/runsync")

    def test_model_env_is_read_at_construction_time(self):
        with patch.dict(os.environ, {"DEFAULT_MODEL_NAME": "env-model"}):
            llm = CustomLLM(system_message="s")
        self.assertEqual(llm.model, "env-model")

    def test_default_max_retries_is_three(self):
        self.assertEqual(CustomLLM(system_message="s").max_retries, 3)

    def test_timeout_constructor_param(self):
        self.assertEqual(CustomLLM(system_message="s", timeout=5).timeout, 5)

    def test_timeout_env_is_read_at_construction_time(self):
        with patch.dict(os.environ, {"LLM_TIMEOUT": "7"}):
            self.assertEqual(CustomLLM(system_message="s").timeout, 7)


class TestRaiseOnEmpty(unittest.TestCase):
    _EMPTY = {"output": {"response": ""}}

    def test_constructor_flag_raises_on_empty_completion(self):
        llm = CustomLLM(system_message="s", raise_on_empty=True)
        with patch.object(llms, "urlopen", return_value=_FakeResponse(self._EMPTY)):
            with self.assertRaises(RuntimeError):
                llm("hello")

    def test_per_call_flag_raises_on_empty_completion(self):
        llm = CustomLLM(system_message="s")
        with patch.object(llms, "urlopen", return_value=_FakeResponse(self._EMPTY)):
            with self.assertRaises(RuntimeError):
                llm("hello", raise_on_empty=True)

    def test_per_call_flag_can_disable_constructor_flag(self):
        llm = CustomLLM(system_message="s", raise_on_empty=True)
        with patch.object(llms, "urlopen", return_value=_FakeResponse(self._EMPTY)):
            self.assertEqual(llm("hello", raise_on_empty=False), "")


class TestAgentImprovements(unittest.TestCase):
    def test_agent_reuses_one_llm_across_calls(self):
        agent = Agent(system_message="s")
        held = agent.llm
        with patch.object(llms, "CustomLLM") as mock_cls:
            with patch.object(
                llms, "urlopen", return_value=_FakeResponse(_OK_BODY)
            ):
                agent("hi")
        mock_cls.assert_not_called()
        self.assertIs(agent.llm, held)

    def test_agent_input_state_is_removed(self):
        with self.assertRaises(TypeError):
            Agent(input={"k": "v"})
        self.assertFalse(hasattr(Agent(), "input"))

    def test_agent_call_url_override(self):
        agent = Agent(system_message="s")
        with patch.object(
            llms, "urlopen", return_value=_FakeResponse(_OK_BODY)
        ) as mock_open:
            agent("p", url="http://other.example/runsync")
        self.assertEqual(
            mock_open.call_args[0][0].full_url, "http://other.example/runsync"
        )

    def test_agent_forwards_llm_kwargs(self):
        agent = Agent(system_message="s", timeout=5, raise_on_empty=True)
        self.assertEqual(agent.llm.timeout, 5)
        self.assertTrue(agent.llm.raise_on_empty)

    def test_agent_stream_delegates_to_llm(self):
        agent = Agent(system_message="s")
        responses = [
            _FakeResponse({"id": "job-2"}),
            _FakeResponse({"status": "COMPLETED",
                           "stream": [{"output": {"response": "chunk"}}]}),
        ]
        with patch.object(llms, "_sleep"):
            with patch.object(llms, "urlopen", side_effect=responses):
                self.assertEqual(list(agent.stream("hello")), ["chunk"])


class TestProductionConstraints(unittest.TestCase):
    """Project rules: stdlib urllib only, no requests/httpx/langchain."""

    def _source(self):
        return open(llms.__file__).read()

    def test_no_requests_or_httpx_import(self):
        src = self._source()
        self.assertNotIn("import requests", src)
        self.assertNotIn("import httpx", src)

    def test_no_langchain_import(self):
        self.assertNotIn("langchain", self._source())


if __name__ == "__main__":
    unittest.main()
