"""One harness protocol for all checkpoint variants and benchmark tasks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

from ..common import (
    add_model_arguments,
    local_checkpoint_metadata,
    package_versions,
    result_path,
    run_metadata,
    save_json,
    validate_model_name,
    validate_model_path,
)


TASKS: dict[str, dict[str, Any]] = {
    "mmlu_pro": {
        "dataset": "TIGER-Lab/MMLU-Pro",
        "dataset_config": None,
        "split": "test",
        "task_identifier": "mmlu_pro",
        "primary_metric": "exact_match,custom-extract",
        "metric_label": "accuracy",
    },
    "gsm8k": {
        "dataset": "openai/gsm8k",
        "dataset_config": "main",
        "split": "test",
        "task_identifier": "gsm8k",
        "primary_metric": "exact_match,strict-match",
        "metric_label": "exact_match",
    },
    "hellaswag": {
        "dataset": "Rowan/hellaswag",
        "dataset_config": None,
        "split": "validation",
        "task_identifier": "hellaswag",
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
    parser.set_defaults(output_dir=Path("results/quality/benchmarks"))
    parser.add_argument("--batch-size", type=validate_batch_size, default="auto")
    parser.add_argument("--limit", type=int, help="Examples per task; engineering runs only.")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _score(metrics: Mapping[str, Any], metric: str) -> float:
    if metric not in metrics:
        raise ValueError(f"Harness result is missing required metric {metric!r}.")
    return float(metrics[metric])


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
    if benchmark == "hellaswag":
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
) -> dict[str, Any]:
    spec = TASKS[benchmark]
    return {
        "evaluation": "benchmark",
        "benchmark": benchmark,
        **run_metadata(model_name, model_path, device),
        **local_checkpoint_metadata(model_path),
        "dataset": spec["dataset"],
        "dataset_config": spec["dataset_config"],
        "split": spec["split"],
        "task_identifier": spec["task_identifier"],
        "num_fewshot": _fewshot_count(raw_results, benchmark),
        "fewshot_protocol": "official harness task default",
        "seed": seed,
        "limit": limit,
        "batch_size": batch_size,
        "chat_template_applied": False,
        "thinking_setting": None,
        "generation_settings": "official harness task default",
        "primary_metric": spec["primary_metric"],
        "metric_label": spec["metric_label"],
        **extract_scores(benchmark, raw_results),
        "raw_results": raw_results,
        "package_versions": package_versions("lm-eval", "torch", "transformers", "datasets"),
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
    seed: int = 42,
) -> dict[str, Any]:
    if benchmark not in TASKS:
        raise ValueError(f"Unknown benchmark: {benchmark}.")
    validate_model_name(model_name)
    validate_model_path(model_path)
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive example count.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    batch_size = validate_batch_size(str(batch_size))
    # Task YAML supplies prompt, few-shot count, filters, and generation settings.
    # These arguments are identical for BF16 and both compressed checkpoints.
    import lm_eval

    raw_results = lm_eval.simple_evaluate(
        model="hf",
        model_args={
            "pretrained": model_path,
            "dtype": "auto",
            "trust_remote_code": False,
        },
        tasks=[TASKS[benchmark]["task_identifier"]],
        num_fewshot=None,
        batch_size=batch_size,
        device=device,
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
        benchmark, model_path, model_name, device, batch_size, limit, seed, raw_results
    )
    save_json(result, result_path(Path(output_dir) / benchmark, model_name))
    return result
