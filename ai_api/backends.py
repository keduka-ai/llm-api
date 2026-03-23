import logging

import requests

import config

logger = logging.getLogger("octagent-backends")


class BackendRegistry:
    """Maps model names to their llama backend container URLs."""

    def __init__(self):
        self.backends = {}
        if config.MODEL1_NAME and config.MODEL1_URL:
            self.backends[config.MODEL1_NAME] = config.MODEL1_URL
        if config.MODEL2_NAME and config.MODEL2_URL:
            self.backends[config.MODEL2_NAME] = config.MODEL2_URL
        self.default_model = config.MODEL1_NAME or None

    def get_backend_url(self, model_name=None):
        """Return the backend URL for the given model name.

        Falls back to default (MODEL1) if model_name is None or empty.
        Raises ValueError if model not found.
        """
        if not model_name:
            model_name = self.default_model
        url = self.backends.get(model_name)
        if not url:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Available: {list(self.backends.keys())}"
            )
        return url

    def get_all_models(self):
        """Return list of all registered model names."""
        return list(self.backends.keys())

    def get_all_backends(self):
        """Return dict of model_name -> url."""
        return dict(self.backends)

    def check_health(self, timeout=10):
        """Check health of all backends by hitting their /v1/models endpoint."""
        results = {}
        for name, url in self.backends.items():
            try:
                resp = requests.get(f"{url}/v1/models", timeout=timeout)
                if resp.status_code == 200:
                    results[name] = {"status": "ok"}
                else:
                    results[name] = {
                        "status": "unhealthy",
                        "http_status": resp.status_code,
                    }
            except requests.RequestException as e:
                logger.warning(f"Backend '{name}' at {url} unreachable: {e}")
                results[name] = {"status": "unreachable", "error": str(e)}
        return results
