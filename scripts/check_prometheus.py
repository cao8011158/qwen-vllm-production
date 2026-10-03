"""Validate Prometheus readiness, scrape target health and real metric data."""

from qwen_vllm_production.observability.cli import main

if __name__ == "__main__":
    main("prometheus")
