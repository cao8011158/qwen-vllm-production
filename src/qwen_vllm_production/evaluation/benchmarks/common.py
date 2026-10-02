"""One harness protocol for all checkpoint variants and benchmark tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from ..common import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_SEED,
    DEFAULT_TENSOR_PARALLEL_SIZE,
    add_model_arguments,
    local_checkpoint_metadata,
    package_versions,
    result_path,
    run_metadata,
    save_json,
    validate_model_name,
    validate_model_path,
    vllm_engine_config,
    vllm_harness_model_args,
)


TASKS: dict[str, dict[str, Any]] = {
    "wikitext": {
        "dataset": "EleutherAI/wikitext_document_level",
        "dataset_config": "wikitext-2-raw-v1",
        "split": "test",
        "task_identifier": "wikitext",
        "output_type": "loglikelihood_rolling",
        "primary_metric": "word_perplexity",
        "metric_label": "word_perplexity",
    },
    "mmlu_pro": {
        "dataset": "TIGER-Lab/MMLU-Pro",
        "dataset_config": None,
        "split": "test",
        "task_identifier": "mmlu_pro",
        "output_type": "generate_until",
        "primary_metric": "exact_match,custom-extract",
        "metric_label": "accuracy",
    },
    "gsm8k": {
        "dataset": "openai/gsm8k",
        "dataset_config": "main",
        "split": "test",
        "task_identifier": "gsm8k",
        "output_type": "generate_until",
        "primary_metric": "exact_match,strict-match",
        "metric_label": "exact_match",
    },
    "hellaswag": {
        "dataset": "Rowan/hellaswag",
        "dataset_config": None,
        "split": "validation",
        "task_identifier": "hellaswag",
        "output_type": "multiple_choice",
        "primary_metric": "acc_norm,none",
        "metric_label": "acc_norm",
    },
}

MMLU_SUBJECTS = (
    "biology", "business", "chemistry", "computer_science", "economics",
    "engineering", "health", "history", "law", "math", "other",
    "philosophy", "physics", "psychology",
)


def validate_batch_size(value: str) -> int | str:
    if value == "auto":
        return value
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("batch-size must be a positive integer or auto.") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("batch-size must be a positive integer or auto.")
    return parsed


def benchmark_parser(benchmark: str) -> argparse.ArgumentParser:
    if benchmark not in TASKS:
        raise ValueError(f"Unknown benchmark: {benchmark}.")
    parser = argparse.ArgumentParser(description=f"Run the official {benchmark} task.")
    add_model_arguments(parser)
    default_output = (
        Path("results/quality/perplexity") if benchmark == "wikitext"
        else Path("results/quality/benchmarks")
    )
    parser.set_defaults(output_dir=default_output)
    parser.add_argument("--batch-size", type=validate_batch_size, default="auto")
    parser.add_argument(
        "--limit", type=int,
        help="Harness task documents/examples, not tokens; engineering runs only.",
    )
    return parser


def _score(metrics: Mapping[str, Any], metric: str) -> float:
    key = metric if metric in metrics else f"{metric},none"
    if key not in metrics:
        raise ValueError(f"Harness result is missing required metric {metric!r}.")
    return float(metrics[key])


def _fewshot_count(raw_results: Mapping[str, Any], task: str) -> int | None:
    counts = raw_results.get("n-shot", {})
    if task in counts:
        return counts[task]
    if task == "mmlu_pro":
        subject_counts = {
            counts[name] for name in counts if name.startswith("mmlu_pro_")
        }
        if len(subject_counts) == 1:
            return subject_counts.pop()
    return None


def extract_scores(benchmark: str, raw_results: Mapping[str, Any]) -> dict[str, Any]:
    spec = TASKS[benchmark]
    results = raw_results.get("results", {})
    if benchmark == "mmlu_pro":
        groups = raw_results.get("groups", {})
        group = groups.get("mmlu_pro") or results.get("mmlu_pro")
        if not isinstance(group, Mapping):
            raise ValueError("Harness omitted the full mmlu_pro aggregate result.")
        subject_scores = {
            subject: _score(results[f"mmlu_pro_{subject}"], spec["primary_metric"])
            for subject in MMLU_SUBJECTS
        }
        return {
            "score": _score(group, spec["primary_metric"]),
            "aggregate_score": _score(group, spec["primary_metric"]),
            "subject_scores": subject_scores,
        }
    if benchmark not in results:
        raise ValueError(f"Harness omitted task result {benchmark!r}.")
    task_result = results[benchmark]
    extra: dict[str, Any] = {}
    if benchmark == "wikitext":
        extra["word_perplexity"] = _score(task_result, "word_perplexity")
        for metric in ("byte_perplexity", "bits_per_byte"):
            if metric in task_result or f"{metric},none" in task_result:
                extra[metric] = _score(task_result, metric)
    elif benchmark == "hellaswag":
        extra["acc"] = _score(task_result, "acc,none")
        extra["acc_norm"] = _score(task_result, "acc_norm,none")
    return {"score": _score(task_result, spec["primary_metric"]), **extra}


def build_benchmark_result(
    benchmark: str,
    model_path: str,
    model_name: str,
    device: str,
    batch_size: int | str,
    limit: int | None,
    seed: int,
    raw_results: Mapping[str, Any],
    engine_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    spec = TASKS[benchmark]
    if engine_config is None:
        engine_config = vllm_engine_config(model_path, seed=seed)
    return {
        "evaluation": "perplexity" if benchmark == "wikitext" else "benchmark",
        "backend": "vllm",
        "benchmark": benchmark,
        **run_metadata(model_name, model_path, device),
        **local_checkpoint_metadata(model_path),
        "dataset": spec["dataset"],
        "dataset_config": spec["dataset_config"],
        "split": spec["split"],
        "task_identifier": spec["task_identifier"],
        "output_type": spec["output_type"],
        "num_fewshot": _fewshot_count(raw_results, benchmark),
        "fewshot_protocol": "official harness task default",
        "seed": seed,
        "limit": limit,
        "limit_unit": "documents" if benchmark == "wikitext" else "examples",
        "batch_size": batch_size,
        "tensor_parallel_size": engine_config["tensor_parallel_size"],
        "gpu_memory_utilization": engine_config["gpu_memory_utilization"],
        "max_model_len": engine_config.get("max_model_len"),
        "dtype_configuration": engine_config["dtype"],
        "chat_template_applied": False,
        "thinking_setting": None,
        "generation_settings": "official harness task default",
        "primary_metric": spec["primary_metric"],
        "metric_label": spec["metric_label"],
        **extract_scores(benchmark, raw_results),
        "raw_results": raw_results,
        "package_versions": package_versions(
            "vllm", "lm-eval", "torch", "transformers", "compressed-tensors", "datasets"
        ),
    }


def run_benchmark(
    benchmark: str,
    *,
    model_path: str,
    model_name: str,
    device: str,
    batch_size: int | str = "auto",
    output_dir: Path = Path("results/quality/benchmarks"),
    limit: int | None = None,
    seed: int = DEFAULT_SEED,
    tensor_parallel_size: int = DEFAULT_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len: int | None = None,
) -> dict[str, Any]:
    if benchmark not in TASKS:
        raise ValueError(f"Unknown benchmark: {benchmark}.")
    validate_model_name(model_name)
    validate_model_path(model_path)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive example count.")
    batch_size = validate_batch_size(str(batch_size))
    engine_config = vllm_engine_config(
        model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=seed,
    )
    # Task YAML supplies prompt, few-shot count, filters, and generation settings.
    # These arguments are identical for BF16 and both compressed checkpoints.
    import lm_eval

    raw_results = lm_eval.simple_evaluate(
        model="vllm",
        model_args=vllm_harness_model_args(engine_config),
        tasks=[TASKS[benchmark]["task_identifier"]],
        num_fewshot=None,
        batch_size=batch_size,
        limit=limit,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
        fewshot_random_seed=seed,
        apply_chat_template=False,
        log_samples=False,
    )
    if raw_results is None:
        raise RuntimeError("lm-evaluation-harness returned no results on this process.")
    result = build_benchmark_result(
        benchmark, model_path, model_name, device, batch_size, limit, seed, raw_results,
        engine_config,
    )
    result_dir = Path(output_dir) if benchmark == "wikitext" else Path(output_dir) / benchmark
    save_json(result, result_path(result_dir, model_name))
    return result
