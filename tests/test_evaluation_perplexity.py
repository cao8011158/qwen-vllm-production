"""CPU-only mathematical checks of the sliding-window PPL protocol."""

import math
from pathlib import Path

import pytest

from qwen_vllm_production.evaluation.perplexity import (
    DATASET,
    DATASET_CONFIG,
    SPLIT,
    build_parser,
    masked_labels,
    perplexity_from_nll,
    sliding_window_spans,
)


def test_wikitext_identity() -> None:
    assert (DATASET, DATASET_CONFIG, SPLIT) == (
        "Salesforce/wikitext", "wikitext-2-raw-v1", "test",
    )


def test_windows_score_each_target_once() -> None:
    spans = list(sliding_window_spans(total_tokens=10, max_length=5, stride=3))
    assert spans == [(0, 5, 1), (3, 8, 5), (5, 10, 8)]
    targets = [index for _, end, first in spans for index in range(first, end)]
    assert targets == list(range(1, 10))


def test_short_final_window_does_not_double_count() -> None:
    spans = list(sliding_window_spans(total_tokens=7, max_length=4, stride=2))
    targets = [index for _, end, first in spans for index in range(first, end)]
    assert targets == list(range(1, 7))


def test_masked_labels_hide_overlap() -> None:
    class FakeTokens:
        def __init__(self, values: list[int]) -> None:
            self.values = values

        def clone(self) -> "FakeTokens":
            return FakeTokens(self.values.copy())

        def __setitem__(self, key: tuple[slice, slice], value: int) -> None:
            self.values[key[1]] = [value] * len(self.values[key[1]])

    source = FakeTokens([3, 4, 5, 6, 7])
    labels = masked_labels(source, start=3, first_target=5)
    assert labels.values == [-100, -100, 5, 6, 7]
    assert source.values == [3, 4, 5, 6, 7]


def test_ppl_math() -> None:
    mean_nll, ppl = perplexity_from_nll(6.0, 3)
    assert mean_nll == 2.0
    assert ppl == pytest.approx(math.exp(2.0))


def test_invalid_window_parameters() -> None:
    with pytest.raises(ValueError, match="stride"):
        list(sliding_window_spans(10, 4, 4))


def test_perplexity_parser() -> None:
    args = build_parser().parse_args(
        ["--model-path", "checkpoint", "--model-name", "bf16", "--limit", "20"]
    )
    assert args.limit == 20
    assert args.max_length == 2048
    assert args.stride == 1024
    assert args.output_dir == Path("results/quality/perplexity")
