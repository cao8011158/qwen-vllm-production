"""Small deterministic checkpoint health check using vLLM offline inference."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

from .common import (
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_SEED,
    DEFAULT_TENSOR_PARALLEL_SIZE,
    add_model_arguments,
    local_checkpoint_metadata,
    package_versions,
    result_path,
    run_metadata,
    save_json,
    vllm_engine_config,
)


PROMPTS = (
    "The capital of France is",
    "2 + 2 =",
    "Complete the sentence: The sky is",
)
MAX_NEW_TOKENS = 8
VERSION_PACKAGES = ("vllm", "lm-eval", "torch", "transformers", "compressed-tensors", "datasets")


def extract_generated_samples(outputs: Sequence[Any], prompts: Sequence[str]) -> list[dict[str, str]]:
    """Require one non-empty vLLM continuation for each fixed raw prompt."""
    if len(outputs) != len(prompts):
        raise ValueError("vLLM returned a different number of outputs than prompts.")
    samples = []
    for prompt, output in zip(prompts, outputs):
        if output.prompt != prompt or not output.outputs:
            raise ValueError("vLLM returned a missing or mismatched prompt output.")
        generated_text = output.outputs[0].text
        if not isinstance(generated_text, str) or not generated_text.strip():
            raise ValueError("Sanity generation produced empty text.")
        samples.append({"prompt": prompt, "generated_text": generated_text})
    return samples


def build_sanity_result(
    model_name: str,
    model_path: str,
    device: str,
    *,
    tensor_parallel_size: int = DEFAULT_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len: int | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    engine_config = vllm_engine_config(
        model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=seed,
    )
    return {
        "evaluation": "sanity",
        "backend": "vllm",
        **run_metadata(model_name, model_path, device),
        "engine_load_success": False,
        "generation_success": False,
        "generated_samples": [],
        "model_config_identifier": None,
        "quantization_metadata": None,
        "quantization_method": None,
        "checkpoint_dtype": None,
        "runtime_model_identifier": None,
        "runtime_dtype": None,
        "runtime_quantization": None,
        "runtime_max_model_len": None,
        "tensor_parallel_size": engine_config["tensor_parallel_size"],
        "gpu_memory_utilization": engine_config["gpu_memory_utilization"],
        "max_model_len": engine_config.get("max_model_len"),
        "dtype_configuration": engine_config["dtype"],
        "seed": seed,
        "prompts": list(PROMPTS),
        "generation_settings": {"temperature": 0.0, "max_tokens": MAX_NEW_TOKENS},
        "chat_template_applied": False,
        "thinking_setting": None,
    }


def evaluate(
    model_path: str,
    model_name: str,
    device: str,
    output_dir: Path,
    *,
    tensor_parallel_size: int = DEFAULT_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len: int | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    result = build_sanity_result(
        model_name, model_path, device,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        seed=seed,
    )
    path = result_path(output_dir, model_name)
    try:
        result.update(local_checkpoint_metadata(model_path))
        from vllm import LLM, SamplingParams

        engine = LLM(**vllm_engine_config(
            model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            seed=seed,
        ))
        result["engine_load_success"] = True
        runtime_config = getattr(getattr(engine, "llm_engine", None), "model_config", None)
        if runtime_config is not None:
            for result_key, config_key in (
                ("runtime_model_identifier", "model"),
                ("runtime_dtype", "dtype"),
                ("runtime_quantization", "quantization"),
                ("runtime_max_model_len", "max_model_len"),
            ):
                value = getattr(runtime_config, config_key, None)
                if value is not None:
                    result[result_key] = str(value) if result_key in (
                        "runtime_dtype", "runtime_quantization"
                    ) else value
        sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
        outputs = engine.generate(list(PROMPTS), sampling_params)
        result["generated_samples"] = extract_generated_samples(outputs, PROMPTS)
        result["generation_success"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["package_versions"] = package_versions(*VERSION_PACKAGES)
    save_json(result, path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run vLLM checkpoint sanity checks.")
    add_model_arguments(parser)
    parser.set_defaults(output_dir=Path("results/quality/sanity"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = evaluate(
        args.model_path, args.model_name, args.device, args.output_dir,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        seed=args.seed,
    )
    if not result["generation_success"]:
        raise RuntimeError(result.get("error", "Sanity check failed."))


if __name__ == "__main__":
    main()
