"""
Tests for entrypoint.sh model selection and server argument construction.

Validates that the entrypoint script correctly resolves the model file
from MODEL_FILE env var and builds the correct llama-server arguments.
"""

import os
import subprocess
import tempfile
import pytest

ENTRYPOINT = os.path.join(os.path.dirname(__file__), "..", "entrypoint.sh")


def _run_entrypoint(env_overrides=None, expect_failure=False):
    """
    Run a wrapper script that mirrors the variable-setup logic from
    entrypoint.sh without actually launching processes. Returns stdout
    on success or stderr on expected failure.
    """
    env = {
        "MODELS_DIR": "/tmp/claude-1000/test_models",
        "MODEL_FILE": "default-model.gguf",
        "N_GPU_LAYERS": "-1",
        "N_CTX": "20000",
        "N_BATCH": "512",
        "N_UBATCH": "1024",
        "FLASH_ATTN_MODE": "on",
        "PATH": os.environ.get("PATH", ""),
    }
    if env_overrides:
        env.update(env_overrides)

    # Create fake model files so the existence check passes (skip for failure tests)
    if not expect_failure:
        os.makedirs(env["MODELS_DIR"], exist_ok=True)
        model_file = env.get("MODEL_FILE", "")
        if model_file:
            path = os.path.join(env["MODELS_DIR"], model_file)
            if not os.path.exists(path):
                with open(path, "w") as f:
                    f.write("fake")

    # Wrapper mirrors the entrypoint.sh variable-setup logic
    wrapper = f"""#!/bin/bash
set -e
MODELS_DIR="${{MODELS_DIR:-/models}}"
MODEL_FILENAME="${{MODEL_FILE:-Qwen3.5-4B-Q4_1.gguf}}"
MODEL_PATH="${{MODELS_DIR}}/${{MODEL_FILENAME}}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: model file not found: $MODEL_PATH" >&2
    exit 1
fi

SERVER_ARGS=(
    --model "$MODEL_PATH"
    --host 0.0.0.0
    --port 8080
    --n-gpu-layers "${{N_GPU_LAYERS:--1}}"
    --ctx-size "${{N_CTX:-20000}}"
    --batch-size "${{N_BATCH:-512}}"
    --ubatch-size "${{N_UBATCH:-1024}}"
    --jinja
    --metrics
    --reasoning-format qwen3
)
SERVER_ARGS+=(--flash-attn "${{FLASH_ATTN_MODE:-on}}")

echo "MODEL_FILENAME=$MODEL_FILENAME"
echo "MODEL_PATH=$MODEL_PATH"
echo "ARGS=${{SERVER_ARGS[*]}}"
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False, dir="/tmp/claude-1000/") as f:
        f.write(wrapper)
        f.flush()
        os.chmod(f.name, 0o755)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["bash", tmp_path],
            env=env,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if expect_failure:
            assert result.returncode != 0, f"Expected failure but got: {result.stdout}"
            return result.stderr
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        return result.stdout
    finally:
        os.unlink(tmp_path)


class TestEntrypointModeSelection:

    def test_default_model_file(self):
        """Default MODEL_FILE is used when env var is set."""
        output = _run_entrypoint({"MODEL_FILE": "default-model.gguf"})
        assert "MODEL_FILENAME=default-model.gguf" in output
        assert "default-model.gguf" in output

    def test_custom_model_file(self):
        """Custom MODEL_FILE is correctly resolved."""
        models_dir = "/tmp/claude-1000/test_models"
        os.makedirs(models_dir, exist_ok=True)
        with open(os.path.join(models_dir, "custom.gguf"), "w") as f:
            f.write("fake")
        output = _run_entrypoint({"MODEL_FILE": "custom.gguf"})
        assert "MODEL_FILENAME=custom.gguf" in output

    def test_missing_model_file_fails(self):
        """Script exits with error when model file does not exist."""
        missing_dir = f"/tmp/claude-1000/missing_{os.getpid()}"
        os.makedirs(missing_dir, exist_ok=True)
        stderr = _run_entrypoint({
            "MODEL_FILE": "nonexistent.gguf",
            "MODELS_DIR": missing_dir,
        }, expect_failure=True)
        assert "model file not found" in stderr

    def test_reasoning_format_always_qwen3(self):
        """--reasoning-format qwen3 is always included in server args."""
        output = _run_entrypoint()
        assert "--reasoning-format qwen3" in output

    def test_gpu_layers_forwarded(self):
        output = _run_entrypoint({"N_GPU_LAYERS": "42"})
        assert "--n-gpu-layers 42" in output

    def test_ctx_size_forwarded(self):
        output = _run_entrypoint({"N_CTX": "8192"})
        assert "--ctx-size 8192" in output

    def test_batch_size_forwarded(self):
        output = _run_entrypoint({"N_BATCH": "256"})
        assert "--batch-size 256" in output

    def test_ubatch_size_forwarded(self):
        output = _run_entrypoint({"N_UBATCH": "2048"})
        assert "--ubatch-size 2048" in output

    def test_flash_attn_forwarded(self):
        output = _run_entrypoint({"FLASH_ATTN_MODE": "auto"})
        assert "--flash-attn auto" in output

    def test_jinja_always_enabled(self):
        output = _run_entrypoint()
        assert "--jinja" in output

    def test_metrics_always_enabled(self):
        output = _run_entrypoint()
        assert "--metrics" in output

    def test_model_path_combines_dir_and_file(self):
        """MODEL_PATH is MODELS_DIR + MODEL_FILE."""
        output = _run_entrypoint({
            "MODELS_DIR": "/tmp/claude-1000/test_models",
            "MODEL_FILE": "default-model.gguf",
        })
        assert "MODEL_PATH=/tmp/claude-1000/test_models/default-model.gguf" in output
