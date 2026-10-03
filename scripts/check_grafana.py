"""Validate Grafana health, provisioning and Prometheus datasource queries."""

from qwen_vllm_production.observability.cli import main

if __name__ == "__main__":
    main("grafana")
