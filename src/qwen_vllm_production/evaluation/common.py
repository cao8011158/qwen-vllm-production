"""Shared offline evaluation inputs, model loading, and JSON output."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping


MODEL_NAMES = ("bf16", "int8_w8a8", "awq_w4a16")


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


def model_metadata(model: Any, tokenizer: Any) -> dict[str, Any]:
    config = model.config
    quantization = getattr(config, "quantization_config", None)
    quantization_dict = (
        quantization.to_dict() if hasattr(quantization, "to_dict") else quantization
    )
    return {
        "model_config_identifier": getattr(config, "_name_or_path", None),
        "tokenizer_identifier": getattr(tokenizer, "name_or_path", None),
        "model_vocab_size": getattr(config, "vocab_size", None),
        "quantization_metadata": quantization_dict,
        "quantization_method": (
            quantization_dict.get("quant_method")
            if isinstance(quantization_dict, dict) else None
        ),
        "dtype": str(model.dtype),
        "device": str(model.device),
    }


def local_checkpoint_metadata(model_path: str | Path) -> dict[str, Any]:
    """Read local config metadata without loading weights or contacting a hub."""
    config_file = Path(model_path) / "config.json"
    if not config_file.is_file():
        return {
            "model_config_identifier": None,
            "quantization_metadata": None,
            "quantization_method": None,
        }
    config = json.loads(config_file.read_text(encoding="utf-8"))
    quantization = config.get("quantization_config")
    return {
        "model_config_identifier": config.get("_name_or_path"),
        "quantization_metadata": quantization,
        "quantization_method": (
            quantization.get("quant_method") if isinstance(quantization, dict) else None
        ),
    }


def load_model_and_tokenizer(model_path: str | Path, device: str) -> tuple[Any, Any]:
    """Use one loader for all variants; checkpoint metadata controls decoding."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    identifier = validate_model_path(model_path)
    tokenizer = AutoTokenizer.from_pretrained(identifier, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        identifier, torch_dtype="auto", device_map=device, trust_remote_code=False
    )
    model.eval()
    return model, tokenizer


def input_device(model: Any) -> Any:
    return model.get_input_embeddings().weight.device


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", required=True, choices=MODEL_NAMES)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path)
