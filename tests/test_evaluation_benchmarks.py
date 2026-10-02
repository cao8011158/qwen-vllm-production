"""Mock-only checks of the shared official-harness protocol."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_vllm_production.evaluation.benchmarks import gsm8k, hellaswag, mmlu_pro
from qwen_vllm_production.evaluation.benchmarks.common import (
    MMLU_SUBJECTS,
    TASKS,
    build_benchmark_result,
    extract_scores,
    run_benchmark,
)


def test_official_task_mapping() -> None:
    assert TASKS["mmlu_pro"]["dataset"] == "TIGER-Lab/MMLU-Pro"
    assert TASKS["mmlu_pro"]["split"] == "test"
    assert TASKS["mmlu_pro"]["task_identifier"] == "mmlu_pro"
    assert len(MMLU_SUBJECTS) == 14
    assert TASKS["gsm8k"]["dataset"] == "openai/gsm8k"
    assert TASKS["gsm8k"]["dataset_config"] == "main"
    assert TASKS["gsm8k"]["split"] == "test"
    assert TASKS["hellaswag"]["dataset"] == "Rowan/hellaswag"
    assert TASKS["hellaswag"]["split"] == "validation"


def test_mmlu_pro_group_accuracy_and_subjects() -> None:
    raw = {
        "groups": {"mmlu_pro": {"exact_match,custom-extract": 0.60}},
        "results": {
            f"mmlu_pro_{subject}": {"exact_match,custom-extract": 0.60}
            for subject in MMLU_SUBJECTS
        },
    }
    scores = extract_scores("mmlu_pro", raw)
    assert scores["aggregate_score"] == 0.60
    assert len(scores["subject_scores"]) == 14


def test_gsm8k_strict_primary_keeps_other_filters() -> None:
    raw = {"results": {"gsm8k": {
        "exact_match,strict-match": 0.40,
        "exact_match,flexible-extract": 0.50,
    }}}
    result = build_benchmark_result(
        "gsm8k", "model", "bf16", "cuda", "auto", None, 42, raw
    )
    assert result["primary_metric"] == "exact_match,strict-match"
    assert result["score"] == 0.40
    assert result["raw_results"] is raw
    assert result["raw_results"]["results"]["gsm8k"]["exact_match,flexible-extract"] == 0.50


def test_hellaswag_acc_and_acc_norm() -> None:
    raw = {"results": {"hellaswag": {"acc,none": 0.45, "acc_norm,none": 0.55}}}
    scores = extract_scores("hellaswag", raw)
    assert scores == {"score": 0.55, "acc": 0.45, "acc_norm": 0.55}


@pytest.mark.parametrize("module", [mmlu_pro, gsm8k, hellaswag])
def test_benchmark_parsers(module: object) -> None:
    args = module.build_parser().parse_args([
        "--model-path", "checkpoint", "--model-name", "int8_w8a8",
        "--device", "cuda", "--batch-size", "4", "--limit", "20", "--seed", "7",
    ])
    assert args.batch_size == 4
    assert args.limit == 20
    assert args.seed == 7
    assert args.output_dir == Path("results/quality/benchmarks")


def test_shared_runner_has_no_variant_scoring_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []
    raw = {
        "results": {"gsm8k": {"exact_match,strict-match": 0.25,
                              "exact_match,flexible-extract": 0.30}},
        "n-shot": {"gsm8k": 5},
        "configs": {"gsm8k": {"num_fewshot": 5}},
    }

    def fake_evaluate(**kwargs: object) -> dict:
        calls.append(dict(kwargs))
        return raw

    monkeypatch.setitem(sys.modules, "lm_eval", SimpleNamespace(simple_evaluate=fake_evaluate))
    for name in ("bf16", "int8_w8a8", "awq_w4a16"):
        result = run_benchmark(
            "gsm8k", model_path=str(tmp_path / name), model_name=name,
            device="cuda", batch_size="auto", output_dir=tmp_path,
            limit=20, seed=7,
        )
        assert result["num_fewshot"] == 5
        assert result["score"] == 0.25
        assert result["limit"] == 20
        assert result["seed"] == 7
        assert result["batch_size"] == "auto"
        saved = json.loads((tmp_path / "gsm8k" / f"{name}.json").read_text(encoding="utf-8"))
        assert saved["raw_results"] == raw
    normalized = [{key: value for key, value in call.items() if key != "model_args"}
                  for call in calls]
    assert normalized[0] == normalized[1] == normalized[2]
    assert all(call["tasks"] == ["gsm8k"] for call in calls)
    assert all(call["num_fewshot"] is None for call in calls)
    assert all(call["apply_chat_template"] is False for call in calls)
    assert all(call["random_seed"] == 7 for call in calls)


def test_invalid_limit_rejected_before_harness() -> None:
    with pytest.raises(ValueError, match="limit"):
        run_benchmark(
            "gsm8k", model_path="checkpoint", model_name="bf16",
            device="cuda", limit=0,
        )
