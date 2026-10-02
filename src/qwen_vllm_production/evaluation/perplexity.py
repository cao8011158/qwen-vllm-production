"""Official WikiText rolling loglikelihood through lm-eval's vLLM backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .benchmarks.common import TASKS, benchmark_parser, run_benchmark
from .common import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_SEED,
    DEFAULT_TENSOR_PARALLEL_SIZE,
)


DATASET = TASKS["wikitext"]["dataset"]
DATASET_CONFIG = TASKS["wikitext"]["dataset_config"]
SPLIT = TASKS["wikitext"]["split"]


def evaluate(
    model_path: str,
    model_name: str,
    device: str,
    output_dir: Path,
    *,
    batch_size: int | str = "auto",
    limit: int | None = None,
    seed: int = DEFAULT_SEED,
    tensor_parallel_size: int = DEFAULT_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len: int | None = None,
) -> dict[str, Any]:
    return run_benchmark(
        "wikitext",
        model_path=model_path,
        model_name=model_name,
        device=device,
        output_dir=output_dir,
        batch_size=batch_size,
        limit=limit,
        seed=seed,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
    )


def build_parser():
    return benchmark_parser("wikitext")


def main() -> None:
    args = build_parser().parse_args()
    evaluate(
        args.model_path, args.model_name, args.device, args.output_dir,
        batch_size=args.batch_size,
        limit=args.limit,
        seed=args.seed,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )


if __name__ == "__main__":
    main()
