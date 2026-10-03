"""Filter the dashboard using the actual server's exported metric families."""

from qwen_vllm_production.observability.metrics import main

if __name__ == "__main__":
    main()
