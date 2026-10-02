"""Small deterministic checkpoint health check, without a benchmark dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .common import (
    add_model_arguments,
    input_device,
    load_model_and_tokenizer,
    model_metadata,
    package_versions,
    result_path,
    run_metadata,
    save_json,
)


PROMPTS = (
    "The capital of France is",
    "2 + 2 =",
    "Complete the sentence: The sky is",
)
MAX_NEW_TOKENS = 8


def logits_are_finite(logits: Any) -> bool:
    import torch

    return bool(torch.isfinite(logits).all().item())


def build_sanity_result(model_name: str, model_path: str, device: str) -> dict[str, Any]:
    return {
        "evaluation": "sanity",
        **run_metadata(model_name, model_path, device),
        "load_success": False,
        "forward_success": False,
        "generation_success": False,
        "finite_logits": False,
        "vocab_size": None,
        "dtype": None,
        "device": None,
        "generated_samples": [],
        "quantization_metadata": None,
        "prompts": list(PROMPTS),
        "generation_settings": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS},
        "chat_template_applied": False,
        "thinking_setting": None,
    }


def evaluate(
    model_path: str, model_name: str, device: str, output_dir: Path
) -> dict[str, Any]:
    import torch

    result = build_sanity_result(model_name, model_path, device)
    path = result_path(output_dir, model_name)
    try:
        model, tokenizer = load_model_and_tokenizer(model_path, device)
        result["load_success"] = True
        result.update(model_metadata(model, tokenizer))
        result["vocab_size"] = tokenizer.vocab_size
        if not isinstance(result["vocab_size"], int) or result["vocab_size"] <= 0:
            raise ValueError("Tokenizer vocabulary size is unavailable or invalid.")
        target_device = input_device(model)
        for prompt in PROMPTS:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {key: value.to(target_device) for key, value in inputs.items()}
            if inputs["input_ids"].shape[-1] < 1:
                raise ValueError("A sanity prompt produced no input tokens.")
            with torch.inference_mode():
                logits = model(**inputs).logits
            if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != inputs["input_ids"].shape[1]:
                raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}.")
            if logits.shape[-1] != model.config.vocab_size:
                raise ValueError("Logits vocabulary dimension differs from model config.")
            if not logits_are_finite(logits):
                raise ValueError("Sanity forward pass produced NaN or Inf logits.")
            result["forward_success"] = True
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, do_sample=False, max_new_tokens=MAX_NEW_TOKENS
                )
            continuation = generated[0, inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
            if not text:
                raise ValueError("Sanity generation produced empty text.")
            result["generated_samples"].append({"prompt": prompt, "generated_text": text})
        result["finite_logits"] = True
        result["generation_success"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["package_versions"] = package_versions("torch", "transformers")
    save_json(result, path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run checkpoint sanity checks.")
    add_model_arguments(parser)
    parser.set_defaults(output_dir=Path("results/quality/sanity"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate(args.model_path, args.model_name, args.device, args.output_dir)
    if not result["generation_success"]:
        raise RuntimeError(result.get("error", "Sanity check failed."))


if __name__ == "__main__":
    main()
