"""Run only the concurrency levels explicitly selected by the operator."""

from qwen_vllm_production.benchmark.runner import main

if __name__ == "__main__":
    main(sweep=True)
