"""Mock-only WikiText task configuration and result checks."""

from pathlib import Path

import pytest

from qwen_vllm_production.evaluation import perplexity
from qwen_vllm_production.evaluation.benchmarks.common import (
    TASKS,
    build_benchmark_result,
    extract_scores,
)


def test_official_wikitext_task_mapping() -> None:
    spec = TASKS["wikitext"]
    assert spec["task_identifier"] == "wikitext"
    assert spec["dataset"] == perplexity.DATASET == "EleutherAI/wikitext_document_level"
    assert spec["dataset_config"] == perplexity.DATASET_CONFIG == "wikitext-2-raw-v1"
    assert spec["split"] == perplexity.SPLIT == "test"
    assert spec["output_type"] == "loglikelihood_rolling"
    assert spec["primary_metric"] == "word_perplexity"


def test_wikitext_primary_and_other_official_metrics() -> None:
    raw = {"results": {"wikitext": {
        "word_perplexity,none": 12.0,
        "byte_perplexity,none": 4.0,
        "bits_per_byte,none": 2.0,
    }}}
    assert extract_scores("wikitext", raw) == {
        "score": 12.0,
        "word_perplexity": 12.0,
        "byte_perplexity": 4.0,
        "bits_per_byte": 2.0,
    }
    result = build_benchmark_result(
        "wikitext", "checkpoint", "bf16", "cuda", "auto", 20, 42, raw,
    )
    assert result["evaluation"] == "perplexity"
    assert result["backend"] == "vllm"
    assert result["primary_metric"] == "word_perplexity"
    assert result["word_perplexity"] == 12.0
    assert result["limit"] == 20
    assert result["limit_unit"] == "documents"
    assert result["raw_results"] is raw


def test_perplexity_parser() -> None:
    args = perplexity.build_parser().parse_args([
        "--model-path", "checkpoint", "--model-name", "bf16",
        "--limit", "20", "--batch-size", "auto", "--tensor-parallel-size", "1",
    ])
    assert args.limit == 20
    assert args.batch_size == "auto"
    assert args.tensor_parallel_size == 1
    assert args.max_model_len is None
    assert args.output_dir == Path("results/quality/perplexity")


def test_perplexity_uses_shared_runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_runner(task: str, **kwargs: object) -> dict:
        captured.update({"task": task, **kwargs})
        return {"word_perplexity": 12.0}

    monkeypatch.setattr(perplexity, "run_benchmark", fake_runner)
    result = perplexity.evaluate(
        "checkpoint", "bf16", "cuda", tmp_path,
        batch_size=4, limit=20, seed=7, max_model_len=4096,
    )
    assert result == {"word_perplexity": 12.0}
    assert captured["task"] == "wikitext"
    assert captured["limit"] == 20
    assert captured["batch_size"] == 4
    assert captured["seed"] == 7
    assert captured["max_model_len"] == 4096
