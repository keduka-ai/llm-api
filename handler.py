"""Root-level entry point for RunPod handler discovery and GitHub-based deployment."""

import runpod
from src.handler import handler  # noqa: F401

runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
