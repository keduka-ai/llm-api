"""
Static configuration for the Gemma RunPod serverless worker.

The handler (`src/handler.py`) reads its runtime knobs straight from the
environment; this module is the single place that records per-model context
sizes and GPU defaults for reference by the entrypoint and any tooling. It
intentionally has no third-party imports so it is safe to import in tests.
"""

import os
from pathlib import Path

APPLICATION = "gemma"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEBUG = os.environ.get("DEBUG", "0") == "1"

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")

# ---------------------------------------------------------------------------
# Per-model configuration, keyed by GGUF filename.
# ---------------------------------------------------------------------------
MODEL_CONFIG = {
    "gemma-4-E2B-it-UD-Q6_K_XL.gguf": {"n_ctx": 40_192, "n_ubatch": 1024},
    "gemma-4-E2B-it-UD-Q4_K_XL.gguf": {"n_ctx": 40_192, "n_ubatch": 1024},
}
DEFAULT_MODEL_CONFIG = {"n_ctx": 40_192, "n_ubatch": 1024}


def get_model_config(model_path_str):
    """Look up MODEL_CONFIG by the GGUF filename from a model path."""
    return MODEL_CONFIG.get(os.path.basename(model_path_str), DEFAULT_MODEL_CONFIG)


# ---------------------------------------------------------------------------
# GPU configuration (entrypoint reads the matching env vars directly).
# ---------------------------------------------------------------------------
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", -1))
N_BATCH = int(os.environ.get("N_BATCH", 512))
N_UBATCH = int(os.environ.get("N_UBATCH", 1024))
MAIN_GPU = int(os.environ.get("MAIN_GPU", 0))

# ---------------------------------------------------------------------------
# Generation defaults — kept aligned with src/handler.py and the reference
# RunPod service so the job I/O contract stays backward compatible (R9.3).
# ---------------------------------------------------------------------------
DEFAULT_MODEL_LABEL = os.environ.get("DEFAULT_MODEL_NAME", "gemma-4-e2b-it")
MAX_GENERATION_TOKENS = int(os.environ.get("MAX_GENERATION_TOKENS", 40_192))
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", 4096))
DEFAULT_SYSTEM_PROMPT = os.environ.get(
    "DEFAULT_SYSTEM_PROMPT",
    "You are a highly knowledgeable, kind, and helpful assistant.",
)
