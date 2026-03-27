from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

APPLICATION = "octagent"

PROJECT_ROOT = Path(__file__).parent.parent

DEBUG = bool(int(os.environ.get("DEBUG", 0)))

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
MODEL_CONFIG = {
    "Qwen3.5-9B-UD-Q4_K_XL.gguf": {"n_ctx": 20_000, "chat_format": None, "n_ubatch": 1024},
    "Qwen3.5-4B-Q4_1.gguf": {"n_ctx": 20_000, "chat_format": None, "n_ubatch": 1024},
}

# Default config used when the model filename is not in MODEL_CONFIG
DEFAULT_MODEL_CONFIG = {"n_ctx": 20_000, "chat_format": None, "n_ubatch": 1024}

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")


def get_model_config(model_path_str):
    """Look up MODEL_CONFIG by the GGUF filename from a model path."""
    filename = os.path.basename(model_path_str)
    return MODEL_CONFIG.get(filename, DEFAULT_MODEL_CONFIG)


# ---------------------------------------------------------------------------
# GPU configuration
# ---------------------------------------------------------------------------
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", -1))
N_BATCH = int(os.environ.get("N_BATCH", 512))
N_UBATCH = int(os.environ.get("N_UBATCH", 512))
FLASH_ATTN = bool(int(os.environ.get("FLASH_ATTN", 1)))
USE_MMAP = bool(int(os.environ.get("USE_MMAP", 1)))
USE_MLOCK = bool(int(os.environ.get("USE_MLOCK", 1)))
MAIN_GPU = int(os.environ.get("MAIN_GPU", 0))

_tensor_split_raw = os.environ.get("TENSOR_SPLIT", "")
TENSOR_SPLIT = [float(x) for x in _tensor_split_raw.split(",") if x.strip()] or None

# ---------------------------------------------------------------------------
# Generation defaults
# ---------------------------------------------------------------------------
MAX_GENERATION_TOKENS = int(os.environ.get("MAX_GENERATION_TOKENS", 75_000))
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", 4096))
DEFAULT_GENERATION_MAX_TOKENS = MAX_GENERATION_TOKENS
DEFAULT_SYSTEM_PROMPT = "You are a highly knowledgeable, kind, and helpful assistant."
