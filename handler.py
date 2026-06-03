"""Root-level entry point for RunPod handler discovery and GitHub-based deployment.

RunPod looks for a top-level `handler.py`. The real logic lives in
`src/handler.py`; this module just wires it into the serverless runtime.
"""

import runpod

from src.handler import handler  # noqa: F401

runpod.serverless.start({"handler": handler, "return_aggregate_stream": True})
