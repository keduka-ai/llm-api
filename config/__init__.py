from pathlib import Path
import sys
import os
from dotenv import load_dotenv

load_dotenv()

APPLICATION = "octagent"

# Define project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Get debug value from environmenthttp://127.0.0.1:9080/kais/dashboard/
DEBUG = bool(int(os.environ.get("DEBUG", 0)))

LOG_DIR = Path().home() / "logs"
# LOG_DIR = PROJECT_ROOT / "logs"
# LOG_DIR.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PROJECT_ROOT))

# Define location of models
MODELS_PATH = Path().home() / "models"
# MODELS_PATH.mkdir(parents=True, exist_ok=True)

# Define location of models
FULL_MODELS_PATH = Path().home() / "models" / "phi3medium-model_original"
# FULL_MODELS_PATH.mkdir(parents=True, exist_ok=True)

# Directory to store test scripts
TEST_DIR = PROJECT_ROOT / "tests"
# TEST_DIR.mkdir(parents=True, exist_ok=True)

INDEXES_FOLDER = PROJECT_ROOT / "indexes"
# INDEXES_FOLDER.mkdir(parents=True, exist_ok=True)

# Define location of tokenizer
TOKENIZER_PATH = Path().home() / "tokenizer"
# TOKENIZER_PATH.mkdir(parents=True, exist_ok=True)

# Set the maximum number of log files to keep, default: 90.
MAX_LOG_FILES = 90

# Model id in huggingface to use
hub_model_name = "microsoft/Phi-3-medium-128k-instruct"
SAVE_MODELS_PATH = MODELS_PATH / "phi3medium-model"
SAVE_TOKENIZER_PATH = TOKENIZER_PATH / "phi3medium-tokenizer"

# Define quantizer path
QUANTIZER_PATH = PROJECT_ROOT / "llama.cpp/build/bin/llama-quantize"

# Define path to hf converter
CONVERTER_PATH = PROJECT_ROOT / "llama.cpp/convert_hf_to_gguf.py"

# Default context window (n_ctx) per model type (fallback when model not in MODEL_CONFIG)
DEFAULT_MAX_TOKENS = {
    "instruct": 90_000,
    "reasoning": 70_000,
}


MODEL_CONFIG = {
    # --- Instruct models ---
    "Qwen3.5-4B-Q4_1.gguf": {"n_ctx": 20_000, "chat_format": None, "n_ubatch": 1024},
    # --- Reasoning models ---
    "Phi-4-mini-reasoning-Q4_K_M.gguf": {"n_ctx": 10_000, "chat_format": None, "n_ubatch": 1024},
}


def get_model_config(model_path_str):
    """Look up MODEL_CONFIG by the GGUF filename from a model path."""
    filename = os.path.basename(model_path_str)
    return MODEL_CONFIG.get(filename, {})


N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", -1))

# Multi-GPU: split model across GPUs (comma-separated floats, e.g. "0.5,0.5")
_tensor_split_raw = os.environ.get("TENSOR_SPLIT", "")
TENSOR_SPLIT = [float(x) for x in _tensor_split_raw.split(",") if x.strip()] or None
MAIN_GPU = int(os.environ.get("MAIN_GPU", 0))

# Inference performance
N_BATCH = int(os.environ.get("N_BATCH", 512))
N_UBATCH = int(os.environ.get("N_UBATCH", 512))
USE_MMAP = bool(int(os.environ.get("USE_MMAP", 1)))
USE_MLOCK = bool(int(os.environ.get("USE_MLOCK", 1)))
FLASH_ATTN = bool(int(os.environ.get("FLASH_ATTN", 1)))

# When set, load only this model (e.g. "instruct" or "reasoning").
# When empty, skip model loading (gateway/proxy mode).
ACTIVE_MODEL = os.environ.get("ACTIVE_MODEL", "")

# Initialize the model path
model_paths = {
    # --- Instruct models (pick one) ---
    "instruct": "ai_api/models/Qwen3.5-4B-Q4_1.gguf", 

    # --- Reasoning models (pick one) ---
    "reasoning": "ai_api/models/Phi-4-mini-reasoning-Q4_K_M.gguf",

}

# Maximum number of llm workers.
# This defines the number of workers attempting the same propmt.
MAX_LLM_WORKERS = 5
MAX_LLM_WORKERS_NOT_COMPLEX = 2

# maximum system prompt tokens
MAX_TOKENS_SYS_PROMPT = 2048
# temperature value for system agent
SYSTEM_AGENT_TEMP = 1e-15
# Set default Max tokens
DEFAULT_MAX_TOKENS_WORKER = 8000
DEFAULT_GENERATION_MAX_TOKENS=75_000

# Ebedding model settings
EMBEDDING_URL = os.environ.get("EMBEDDING_URL")
D_MODEL = 768  # dimensions of embedding model
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
DEFAULT_LLM_MODEL_NAME = "instruct"
DEFAULT_CODER_MODEL_NAME = "qwen2_coder"
DEFAULT_TEMPERATURE = 1e-8

# Minimum ratio of workers with code before execution
MIN_CODE_EXEC = 0.3

# Backend model routing (for separate llama-cpp-python server containers)
MODEL1_NAME = os.environ.get("MODEL1_NAME", "")
MODEL1_URL = os.environ.get("MODEL1_URL", "")
MODEL2_NAME = os.environ.get("MODEL2_NAME", "")
MODEL2_URL = os.environ.get("MODEL2_URL", "")

LLM_URL = os.environ.get("LLM_URL")

SYSTEM_AGENT_URL = os.environ.get("SYSTEM_AGENT_URL")

PYAGENT_URL = os.environ.get("PYAGENT_URL")

# Do you want to execute python code to do computation
EXEC_PYTHON_CODE = False

# Path to save default index
# Thresholds above .9 works best for `distance_measure_algos` (lower is better)
# thresholds below .45 works best for `similarity_measure_algos` (higher is better)
INDEXING_ALGORITHM = "create_faiss_index_flat_ip"
DEFAULT_INDEX = INDEXES_FOLDER / "flat_ivf_scalar_quantizer.bin"
DEFAULT_NEAREST_NEIGHBOR = 2
DEFAULT_NEAREST_NEIGHBOR_THRESHOLD = 0.65  # Lower is better (0, 2)
MAX_NEAREST_NEIGHBOR = DEFAULT_NEAREST_NEIGHBOR + int(DEFAULT_NEAREST_NEIGHBOR / 2)

# Include context from repo
DEFAULT_NEAREST_NEIGHBOR_REPO_CONTEXT = 3
DEFAULT_NEAREST_NEIGHBOR_THRESHOLD_REPO_CONTEXT = 0.65
ENHANCED_CONTEXT = True

# Configuration values for local project
DEFAULT_NEAREST_NEIGHBOR_THRESHOLD_LOCAL_PROJECT = 0.3

# Configuration for indexing algorithm
NLIST = 100  # Number of clusters to partition vectors
NPROBE = 10  # The number of clusters to search

# Minimum word count for context returned
MIN_CONTEXT_WC = 50

# repo_paths = "/home/samusachi/WorkStation/ai-services/agents/repo_records/repos.json"

# Minimum number of repo records required
# Records include general python code snipets, python code from repos, and markdown docs.
# Each should have the minimum count stated below.
MINIMUM_RECORDS = 100000

# Index reset threshold
# What is the size of new records in database to trigger reindexing
REINDEX_VALUE = 10

################################################
REPO_SUMMARY_MAX_TOKEN = 1024
TOPIC_MAX_TOKEN = 128
PLANNER_MAX_TOKEN = 1024
PLANNER_REWRITER_MAX_TOKEN = 1024
