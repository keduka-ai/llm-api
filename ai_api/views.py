from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import config
import threading
import requests as http_requests
from ai_api.backends import BackendRegistry

from custom_logging import setup_logger

logger = setup_logger(
    logger_name=f"{config.APPLICATION}-views",
    log_level="info",
    log_filename=None,
    utc_time=True,
)

logger.info("Running as API gateway — routing to llama-server backends")

# Backend registry for routing to llama-server containers
backend_registry = BackendRegistry()

# Thread-local HTTP sessions for gateway proxy calls (connection pooling)
# requests.Session is not thread-safe; each gateway thread gets its own session.
_thread_local = threading.local()


def _get_proxy_session():
    if not hasattr(_thread_local, "session"):
        session = http_requests.Session()
        adapter = http_requests.adapters.HTTPAdapter(
            pool_connections=4,
            pool_maxsize=4,
        )
        session.mount("http://", adapter)
        _thread_local.session = session
    return _thread_local.session

# Default system prompt message
default_system_prompt = {
    "role": "system",
    "content": "You are a highly knowledgeable, kind, and helpful assistant.",
}


class ChatCompletionsView(APIView):
    """OpenAI-compatible chat completions endpoint.

    Routes requests to llama-server backend containers.
    """

    def post(self, request, *args, **kwargs):
        try:
            messages = request.data.get("messages")
            if not messages:
                return Response(
                    {"error": {"message": "Missing required parameter: 'messages'", "type": "invalid_request_error"}},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            stream = request.data.get("stream", False)
            if stream:
                return Response(
                    {"error": {"message": "Streaming is not supported", "type": "invalid_request_error"}},
                    status=status.HTTP_400_BAD_REQUEST,
                )

            model_name = request.data.get("model", config.DEFAULT_LLM_MODEL_NAME)

            # Route to the correct llama-server backend container
            try:
                backend_url = backend_registry.get_backend_url(model_name)
            except ValueError as e:
                return Response(
                    {"error": {"message": str(e), "type": "invalid_request_error"}},
                    status=status.HTTP_404_NOT_FOUND,
                )

            payload = dict(request.data)
            resp = _get_proxy_session().post(
                f"{backend_url}/v1/chat/completions",
                json=payload,
                timeout=900,
            )

            logger.info(f"ChatCompletions routed to backend={backend_url} status={resp.status_code}")
            try:
                data = resp.json()
            except http_requests.exceptions.JSONDecodeError:
                logger.error(f"Backend returned non-JSON response: {resp.text[:500]}")
                return Response(
                    {"error": {"message": "Backend returned an invalid response", "type": "server_error"}},
                    status=status.HTTP_502_BAD_GATEWAY,
                )

            # Handle thinking content based on whether thinking was requested
            think_requested = payload.get("think", False)
            if resp.status_code == 200:
                for choice in data.get("choices", []):
                    msg = choice.get("message", {})
                    if not think_requested:
                        # Strip reasoning_content and any residual <think> tags
                        msg.pop("reasoning_content", None)
                        content = msg.get("content", "")
                        if "</think>" in content:
                            msg["content"] = content.split("</think>")[-1].strip()

            return Response(data, status=resp.status_code)

        except http_requests.Timeout:
            logger.error("Backend request timed out", exc_info=True)
            return Response(
                {"error": {"message": "Backend request timed out", "type": "server_error"}},
                status=status.HTTP_504_GATEWAY_TIMEOUT,
            )
        except http_requests.ConnectionError:
            logger.error("Backend connection failed", exc_info=True)
            return Response(
                {"error": {"message": "Backend is unreachable", "type": "server_error"}},
                status=status.HTTP_502_BAD_GATEWAY,
            )
        except Exception as e:
            logger.error(f"Error in ChatCompletionsView: {str(e)}", exc_info=True)
            return Response(
                {"error": {"message": "An internal error occurred", "type": "server_error"}},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class TextPromptView(APIView):

    def post(self, request, *args, **kwargs):
        try:
            prompt = request.data.get("prompt", "Hello!\n")
            system_prompt_content = request.data.get(
                "system_prompt", default_system_prompt["content"]
            )
            raw = bool(request.data.get("raw", False))

            # Get the model name from the request or use default
            model_name = request.data.get("model_name", config.DEFAULT_LLM_MODEL_NAME)

            try:
                backend_url = backend_registry.get_backend_url(model_name)
            except ValueError as e:
                return Response(
                    {"error": str(e)},
                    status=status.HTTP_404_NOT_FOUND,
                )

            full_prompt = f"{system_prompt_content}\n\n{prompt}"
            chat_payload = {
                "model": model_name,
                "messages": [{"role": "user", "content": full_prompt}],
                "max_tokens": request.data.get("max_tokens", config.DEFAULT_GENERATION_MAX_TOKENS),
                "temperature": request.data.get("temperature", 0.00005),
                "top_p": request.data.get("top_p", 1.0),
                "repeat_penalty": request.data.get("repeat_penalty", 1.2),
                "think": request.data.get("think", False),
            }

            # Optional params — only forward when explicitly provided
            _optional_params = [
                "top_k", "min_p", "typical_p",
                "presence_penalty", "frequency_penalty",
                "mirostat", "mirostat_tau", "mirostat_eta",
                "seed", "stop", "logit_bias",
                "n_predict", "n_probs",
                "grammar", "json_schema",
            ]
            for key in _optional_params:
                if key in request.data:
                    chat_payload[key] = request.data[key]

            resp = _get_proxy_session().post(
                f"{backend_url}/v1/chat/completions",
                json=chat_payload,
                timeout=900,
            )

            logger.info(f"TextPrompt routed to backend={backend_url} status={resp.status_code}")

            if resp.status_code != 200:
                try:
                    return Response(resp.json(), status=resp.status_code)
                except http_requests.exceptions.JSONDecodeError:
                    logger.error(f"Backend returned non-JSON error: {resp.text[:500]}")
                    return Response(
                        {"error": "Backend returned an invalid response"},
                        status=status.HTTP_502_BAD_GATEWAY,
                    )

            response_data = resp.json()
            if raw:
                return Response(response_data, status=status.HTTP_200_OK)

            response_text = response_data["choices"][0]["message"]["content"]
            if not chat_payload.get("think"):
                response_text = response_text.split("</think>")[-1].strip()
            return Response({"response": response_text}, status=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error in TextPromptView: {str(e)}", exc_info=True)
            return Response(
                {"error": "An error occurred"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
