"""Shared vLLM evaluation inputs, engine settings, and JSON output."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping


MODEL_NAMES = ("bf16", "int8_w8a8", "awq_w4a16")
DEFAULT_TENSOR_PARALLEL_SIZE = 1
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
DEFAULT_SEED = 42
DTYPE_CONFIGURATION = "auto"


def validate_model_name(name: str) -> str:
    if name not in MODEL_NAMES:
        raise ValueError(f"model_name must be one of: {', '.join(MODEL_NAMES)}.")
    return name


def validate_model_path(path: str | Path) -> str:
    value = str(path).strip()
    if not value:
        raise ValueError("model_path must be a non-empty model ID or checkpoint path.")
    return value


def result_path(output_dir: Path, model_name: str) -> Path:
    return Path(output_dir) / f"{validate_model_name(model_name)}.json"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def save_json(payload: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default)
        + "\n",
        encoding="utf-8",
    )


def package_versions(*packages: str) -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for package in packages:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            found[package] = None
    return found


def run_metadata(model_name: str, model_path: str | Path, device: str) -> dict[str, Any]:
    return {
        "model_name": validate_model_name(model_name),
        "model_path": validate_model_path(model_path),
        "requested_device": device,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def local_checkpoint_metadata(model_path: str | Path) -> dict[str, Any]:
    """Read local config metadata without loading weights or contacting a hub."""
    config_file = Path(model_path) / "config.json"
    if not config_file.is_file():
        return {
            "model_config_identifier": None,
            "quantization_metadata": None,
            "quantization_method": None,
            "checkpoint_dtype": None,
        }
    config = json.loads(config_file.read_text(encoding="utf-8"))
    quantization = config.get("quantization_config")
    return {
        "model_config_identifier": config.get("_name_or_path"),
        "quantization_metadata": quantization,
        "quantization_method": (
            quantization.get("quant_method") if isinstance(quantization, dict) else None
        ),
        "checkpoint_dtype": config.get("torch_dtype", config.get("dtype")),
    }


def vllm_engine_config(
    model_path: str | Path,
    *,
    tensor_parallel_size: int = DEFAULT_TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
    max_model_len: int | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Build the same core engine configuration for sanity and lm-eval."""
    if isinstance(tensor_parallel_size, bool) or not isinstance(tensor_parallel_size, int) or tensor_parallel_size <= 0:
        raise ValueError("tensor_parallel_size must be a positive integer.")
    if isinstance(gpu_memory_utilization, bool) or not isinstance(gpu_memory_utilization, (int, float)) or not 0 < gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1].")
    if max_model_len is not None and (isinstance(max_model_len, bool) or not isinstance(max_model_len, int) or max_model_len <= 0):
        raise ValueError("max_model_len must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    config: dict[str, Any] = {
        "model": validate_model_path(model_path),
        "tensor_parallel_size": tensor_parallel_size,
        "dtype": DTYPE_CONFIGURATION,
        "gpu_memory_utilization": float(gpu_memory_utilization),
        "trust_remote_code": False,
        "seed": seed,
    }
    if max_model_len is not None:
        config["max_model_len"] = max_model_len
    return config


def vllm_harness_model_args(engine_config: Mapping[str, Any]) -> dict[str, Any]:
    """lm-eval's vLLM adapter names the engine's model argument 'pretrained'."""
    return {"pretrained": engine_config["model"], **{
        key: value for key, value in engine_config.items() if key != "model"
    }}


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True, choices=MODEL_NAMES)
    parser.add_argument(
        "--device", default="cuda", choices=("cuda",),
        help="Recorded for reproducibility; select the GPU with CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, default=DEFAULT_TENSOR_PARALLEL_SIZE)
    parser.add_argument("--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
