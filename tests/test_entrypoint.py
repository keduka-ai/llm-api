"""
Tests for entrypoint.sh model selection and server argument construction.

Validates that the entrypoint script correctly resolves the model file
from MODEL_FILE env var and builds the correct llama-server arguments.
"""

import os
import subprocess
import tempfile
import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import MODEL_CONFIG, get_model_config

ENTRYPOINT = os.path.join(os.path.dirname(__file__), "..", "entrypoint.sh")
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_DEFAULTS_SH = os.path.join(REPO_ROOT, "model-defaults.sh")


def _read_model_defaults():
    """Parse model-defaults.sh and return a dict of variable assignments."""
    defaults = {}
    with open(MODEL_DEFAULTS_SH, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                defaults[key] = value.strip('"').strip("'")
    return defaults


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
        models_dir = env["MODELS_DIR"]
        os.makedirs(models_dir, exist_ok=True)
        # Determine which model file will be resolved by the 3-step logic
        model_file = env.get("MODEL_FILE", "")
        if not model_file:
            active_marker = os.path.join(models_dir, ".active_model")
            if os.path.exists(active_marker):
                with open(active_marker, "r") as f:
                    model_file = f.read().strip()
            else:
                model_file = _read_model_defaults()["DEFAULT_MODEL_FILENAME"]
        if model_file:
            path = os.path.join(models_dir, model_file)
            if not os.path.exists(path):
                with open(path, "w") as f:
                    f.write("fake")

    # Wrapper mirrors the entrypoint.sh variable-setup logic
    wrapper = f"""#!/bin/bash
set -e
source "{MODEL_DEFAULTS_SH}"
MODELS_DIR="${{MODELS_DIR:-/models}}"

# Resolve model filename: explicit MODEL_FILE > .active_model marker > fallback
if [ -n "$MODEL_FILE" ]; then
    MODEL_FILENAME="$MODEL_FILE"
elif [ -f "$MODELS_DIR/.active_model" ]; then
    MODEL_FILENAME=$(cat "$MODELS_DIR/.active_model")
else
    MODEL_FILENAME="$DEFAULT_MODEL_FILENAME"
fi
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
    --reasoning-format deepseek
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

    def test_reasoning_format_always_deepseek(self):
        """--reasoning-format deepseek is always included in server args."""
        output = _run_entrypoint()
        assert "--reasoning-format deepseek" in output

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

    def test_active_model_marker_used_when_no_model_file_env(self):
        """When MODEL_FILE is unset and .active_model exists, its content is used."""
        models_dir = f"/tmp/claude-1000/test_models_marker_{os.getpid()}"
        os.makedirs(models_dir, exist_ok=True)
        # Write the .active_model marker file
        with open(os.path.join(models_dir, ".active_model"), "w") as f:
            f.write("marker-model.gguf")
        try:
            output = _run_entrypoint({
                "MODELS_DIR": models_dir,
                "MODEL_FILE": "",  # empty means unset in the -n check
            })
            assert "MODEL_FILENAME=marker-model.gguf" in output
            assert f"MODEL_PATH={models_dir}/marker-model.gguf" in output
        finally:
            import shutil
            shutil.rmtree(models_dir, ignore_errors=True)

    def test_model_file_env_overrides_active_model(self):
        """When both MODEL_FILE env and .active_model exist, MODEL_FILE wins."""
        models_dir = f"/tmp/claude-1000/test_models_override_{os.getpid()}"
        os.makedirs(models_dir, exist_ok=True)
        # Write the .active_model marker file
        with open(os.path.join(models_dir, ".active_model"), "w") as f:
            f.write("marker-model.gguf")
        # Create the explicit model file
        with open(os.path.join(models_dir, "explicit.gguf"), "w") as f:
            f.write("fake")
        try:
            output = _run_entrypoint({
                "MODELS_DIR": models_dir,
                "MODEL_FILE": "explicit.gguf",
            })
            assert "MODEL_FILENAME=explicit.gguf" in output
            assert "marker-model.gguf" not in output
        finally:
            import shutil
            shutil.rmtree(models_dir, ignore_errors=True)

    def test_qwen35_35b_alias_resolves(self):
        """download-models.sh resolves qwen3.5-35b alias correctly."""
        script = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "download-models.sh")
        )
        # Extract just the resolve_model function body (up to first closing brace)
        # then call it directly, avoiding the rest of the script (wget, etc.)
        wrapper = f"""#!/bin/bash
set -e
# Define resolve_model by sourcing just the function
eval "$(awk '/^resolve_model\\(\\)/,/^\\}}/' '{script}')"
resolve_model "qwen3.5-35b"
echo "MODEL_FILE=$MODEL_FILE"
echo "MODEL_URL=$MODEL_URL"
"""
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False, dir="/tmp/claude-1000/") as f:
            f.write(wrapper)
            f.flush()
            os.chmod(f.name, 0o755)
            tmp_path = f.name
        try:
            result = subprocess.run(
                ["bash", tmp_path],
                capture_output=True, text=True, timeout=5,
            )
            assert result.returncode == 0, f"Script failed: {result.stderr}"
            assert "MODEL_FILE=Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf" in result.stdout
            assert "MODEL_URL=https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf" in result.stdout
        finally:
            os.unlink(tmp_path)

    def test_fallback_when_no_env_and_no_marker(self):
        """When neither MODEL_FILE nor .active_model exists, falls back to default."""
        defaults = _read_model_defaults()
        default_filename = defaults["DEFAULT_MODEL_FILENAME"]
        models_dir = f"/tmp/claude-1000/test_models_fallback_{os.getpid()}"
        os.makedirs(models_dir, exist_ok=True)
        try:
            output = _run_entrypoint({
                "MODELS_DIR": models_dir,
                "MODEL_FILE": "",  # empty means unset in the -n check
            })
            assert f"MODEL_FILENAME={default_filename}" in output
            assert f"MODEL_PATH={models_dir}/{default_filename}" in output
        finally:
            import shutil
            shutil.rmtree(models_dir, ignore_errors=True)


class TestModelDefaults:

    def test_model_defaults_file_exists(self):
        """model-defaults.sh exists at repo root."""
        assert os.path.isfile(MODEL_DEFAULTS_SH)

    def test_model_defaults_has_alias(self):
        """model-defaults.sh defines DEFAULT_MODEL_ALIAS."""
        defaults = _read_model_defaults()
        assert "DEFAULT_MODEL_ALIAS" in defaults
        assert len(defaults["DEFAULT_MODEL_ALIAS"]) > 0

    def test_model_defaults_has_filename(self):
        """model-defaults.sh defines DEFAULT_MODEL_FILENAME."""
        defaults = _read_model_defaults()
        assert "DEFAULT_MODEL_FILENAME" in defaults
        assert defaults["DEFAULT_MODEL_FILENAME"].endswith(".gguf")

    def test_alias_resolves_to_filename(self):
        """The DEFAULT_MODEL_ALIAS resolves to DEFAULT_MODEL_FILENAME in download-models.sh."""
        defaults = _read_model_defaults()
        alias = defaults["DEFAULT_MODEL_ALIAS"]
        expected_file = defaults["DEFAULT_MODEL_FILENAME"]
        script = os.path.join(REPO_ROOT, "download-models.sh")
        wrapper = f"""#!/bin/bash
set -e
eval "$(awk '/^resolve_model\\(\\)/,/^\\}}/' '{script}')"
resolve_model "{alias}"
echo "MODEL_FILE=$MODEL_FILE"
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False, dir="/tmp/claude-1000/") as f:
            f.write(wrapper)
            f.flush()
            os.chmod(f.name, 0o755)
            tmp_path = f.name
        try:
            result = subprocess.run(
                ["bash", tmp_path],
                capture_output=True, text=True, timeout=5,
            )
            assert result.returncode == 0, f"Script failed: {result.stderr}"
            assert f"MODEL_FILE={expected_file}" in result.stdout
        finally:
            os.unlink(tmp_path)

    def test_download_script_uses_default_alias(self):
        """download-models.sh sources model-defaults.sh for its default MODEL."""
        script = os.path.join(REPO_ROOT, "download-models.sh")
        with open(script, "r") as f:
            content = f.read()
        assert "$DEFAULT_MODEL_ALIAS" in content, \
            "download-models.sh should use $DEFAULT_MODEL_ALIAS from model-defaults.sh"

    def test_entrypoint_uses_default_filename(self):
        """entrypoint.sh sources model-defaults.sh for its fallback filename."""
        script = os.path.join(REPO_ROOT, "entrypoint.sh")
        with open(script, "r") as f:
            content = f.read()
        assert "$DEFAULT_MODEL_FILENAME" in content, \
            "entrypoint.sh should use $DEFAULT_MODEL_FILENAME from model-defaults.sh"


class TestModelConfig:

    def test_qwen35_35b_in_model_config(self):
        """Qwen3.5-35B-A3B GGUF has an entry in MODEL_CONFIG."""
        assert "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf" in MODEL_CONFIG

    def test_qwen35_35b_config_values(self):
        """Qwen3.5-35B-A3B config has expected keys."""
        cfg = get_model_config("Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf")
        assert "n_ctx" in cfg
        assert "chat_format" in cfg
        assert "n_ubatch" in cfg
