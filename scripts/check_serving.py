"""Validate vLLM health, served model and metrics without starting a server."""

from qwen_vllm_production.observability.cli import main

if __name__ == "__main__":
    main("serving")
