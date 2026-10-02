"""Sliding-window causal LM perplexity on WikiText-2 raw test text."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterator

from .common import (
    add_model_arguments,
    input_device,
    load_model_and_tokenizer,
    model_metadata,
    package_versions,
    result_path,
    run_metadata,
    save_json,
    validate_model_name,
    validate_model_path,
)


DATASET = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
SPLIT = "test"
DEFAULT_MAX_LENGTH = 2048
DEFAULT_STRIDE = 1024


def sliding_window_spans(
    total_tokens: int, max_length: int, stride: int
) -> Iterator[tuple[int, int, int]]:
    """Yield (start, end, first_new_target); every target 1..N-1 appears once."""
    if total_tokens < 2:
        raise ValueError("Perplexity requires at least two tokens.")
    if max_length < 2 or not 0 < stride < max_length:
        raise ValueError("max_length must be >= 2 and stride must be in [1, max_length).")
    previous_end = 0
    end = min(max_length, total_tokens)
    while True:
        start = max(0, end - max_length)
        first_target = 1 if previous_end == 0 else previous_end
        yield start, end, first_target
        if end == total_tokens:
            return
        previous_end = end
        end = min(end + stride, total_tokens)


def masked_labels(input_ids: object, start: int, first_target: int) -> object:
    """Mask context tokens; a window's new targets retain their real token IDs."""
    labels = input_ids.clone()
    labels[:, : first_target - start] = -100
    return labels


def perplexity_from_nll(total_nll: float, target_tokens: int) -> tuple[float, float]:
    if target_tokens <= 0 or not math.isfinite(total_nll):
        raise ValueError("Perplexity needs finite NLL and at least one target token.")
    mean_nll = total_nll / target_tokens
    value = math.exp(mean_nll)
    if not math.isfinite(value):
        raise ValueError("Perplexity overflowed; inspect model losses.")
    return mean_nll, value


def evaluate(
    model_path: str,
    model_name: str,
    device: str,
    output_dir: Path,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    stride: int = DEFAULT_STRIDE,
    limit: int | None = None,
) -> dict:
    validate_model_name(model_name)
    validate_model_path(model_path)
    if max_length < 2 or not 0 < stride < max_length:
        raise ValueError("max_length must be >= 2 and stride must be in [1, max_length).")
    if limit is not None and limit < 2:
        raise ValueError("limit must be at least two tokens.")
    import torch
    from datasets import load_dataset

    model, tokenizer = load_model_and_tokenizer(model_path, device)
    dataset = load_dataset(DATASET, DATASET_CONFIG, split=SPLIT)
    text = "\n\n".join(dataset["text"])
    token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
    if limit is not None:
        token_ids = token_ids[:, :limit]
    total_tokens = token_ids.shape[1]
    total_nll = 0.0
    scored_tokens = 0
    target_device = input_device(model)

    for start, end, first_target in sliding_window_spans(total_tokens, max_length, stride):
        window = token_ids[:, start:end].to(target_device)
        labels = masked_labels(window, start, first_target)
        count = end - first_target
        with torch.inference_mode():
            loss = model(input_ids=window, labels=labels).loss
        loss_value = float(loss.float().item())
        if not math.isfinite(loss_value):
            raise ValueError(f"Non-finite loss in token window [{start}, {end}).")
        total_nll += loss_value * count
        scored_tokens += count

    mean_nll, perplexity = perplexity_from_nll(total_nll, scored_tokens)
    if scored_tokens != total_tokens - 1:
        raise RuntimeError("Sliding windows did not score each predictable token once.")
    result = {
        "evaluation": "perplexity",
        **run_metadata(model_name, model_path, device),
        **model_metadata(model, tokenizer),
        "dataset": DATASET,
        "dataset_config": DATASET_CONFIG,
        "split": SPLIT,
        "text_separator": "\\n\\n",
        "chat_template_applied": False,
        "thinking_setting": None,
        "max_length": max_length,
        "stride": stride,
        "limit_tokens": limit,
        "input_tokens": total_tokens,
        "evaluated_tokens": scored_tokens,
        "total_nll": total_nll,
        "mean_nll": mean_nll,
        "perplexity": perplexity,
        "package_versions": package_versions("torch", "transformers", "datasets"),
    }
    save_json(result, result_path(output_dir, model_name))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate WikiText-2 raw perplexity.")
    add_model_arguments(parser)
    parser.set_defaults(output_dir=Path("results/quality/perplexity"))
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--limit", type=int, help="Token limit for an engineering smoke run only.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    evaluate(
        args.model_path, args.model_name, args.device, args.output_dir,
        max_length=args.max_length, stride=args.stride, limit=args.limit,
    )


if __name__ == "__main__":
    main()
