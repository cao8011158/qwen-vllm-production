"""CPU-safe configuration, path, and metadata helpers for compression."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any


Config = dict[str, Any]
_PROFILE_METHODS = {
    "baseline_bf16": "none",
    "int8_w8a8": "int8_w8a8",
    "awq_w4a16": "awq",
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string.")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return value


def load_config(path: Path) -> Config:
    """Read the single YAML configuration; importing this module needs no ML stack."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read configs/config.yaml.") from exc
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except OSError as exc:
        raise ValueError(f"Cannot read configuration at {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    validate_config(config)
    return config


def get_model_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(config.get("model"), "model")


def get_compression_profile(
    config: Mapping[str, Any], profile_name: str
) -> Mapping[str, Any]:
    profiles = _mapping(config.get("compression"), "compression")
    if profile_name not in profiles:
        raise ValueError(f"Unknown compression profile: {profile_name}.")
    return _mapping(profiles[profile_name], f"compression.{profile_name}")


def get_calibration_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return _mapping(config.get("calibration"), "calibration")


def validate_model_config(model: Mapping[str, Any]) -> None:
    _nonempty_string(model.get("model_id"), "model.model_id")
    _nonempty_string(model.get("revision"), "model.revision")
    if model.get("dtype") != "bfloat16":
        raise ValueError("model.dtype must be bfloat16 for this BF16 baseline.")
    if not isinstance(model.get("trust_remote_code"), bool):
        raise ValueError("model.trust_remote_code must be a boolean.")
    _positive_int(model.get("max_model_len"), "model.max_model_len")


def validate_compression_profile(
    profile_name: str, profile: Mapping[str, Any]
) -> None:
    expected_method = _PROFILE_METHODS.get(profile_name)
    if expected_method is None:
        raise ValueError(f"Unknown compression profile: {profile_name}.")
    if profile.get("method") != expected_method:
        raise ValueError(f"compression.{profile_name}.method must be {expected_method}.")
    if profile_name == "baseline_bf16":
        if profile.get("weight_dtype") != "bfloat16" or profile.get("activation_dtype") != "bfloat16":
            raise ValueError("baseline_bf16 must use bfloat16 weights and activations.")
        if profile.get("requires_calibration") is not False:
            raise ValueError("baseline_bf16.requires_calibration must be false.")
    elif profile_name == "int8_w8a8":
        if profile.get("weight_bits") != 8 or profile.get("activation_bits") != 8:
            raise ValueError("int8_w8a8 weight_bits and activation_bits must both be 8.")
        if profile.get("activation_scheme") != "static":
            raise ValueError("int8_w8a8.activation_scheme must be static.")
        if profile.get("requires_calibration") is not True:
            raise ValueError("Static W8A8 requires calibration data.")
    else:
        if profile.get("weight_bits") != 4:
            raise ValueError("awq_w4a16.weight_bits must be 4.")
        if profile.get("activation_dtype") != "bfloat16":
            raise ValueError("awq_w4a16.activation_dtype must be bfloat16.")
        if profile.get("requires_calibration") is not True:
            raise ValueError("awq_w4a16.requires_calibration must be true.")


def validate_calibration_config(
    calibration: Mapping[str, Any], *, required: bool = True
) -> None:
    for key in ("dataset", "split"):
        value = calibration.get(key)
        if value is not None:
            _nonempty_string(value, f"calibration.{key}")
        elif required:
            raise ValueError(f"calibration.{key} must be explicitly configured.")
    for key in ("shuffle", "use_chat_template"):
        value = calibration.get(key)
        if value is not None:
            if not isinstance(value, bool):
                raise ValueError(f"calibration.{key} must be a boolean.")
        elif required:
            raise ValueError(f"calibration.{key} must be explicitly configured.")
    seed = calibration.get("seed")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("calibration.seed must be an integer.")
    elif required:
        raise ValueError("calibration.seed must be explicitly configured.")
    for key in ("num_samples", "max_sequence_length"):
        value = calibration.get(key)
        if value is not None:
            _positive_int(value, f"calibration.{key}")
        elif required:
            raise ValueError(f"calibration.{key} must be explicitly configured.")


def validate_config(config: Mapping[str, Any]) -> None:
    project = _mapping(config.get("project"), "project")
    _nonempty_string(project.get("name"), "project.name")
    _positive_int(project.get("seed"), "project.seed")
    validate_model_config(get_model_config(config))
    profiles = _mapping(config.get("compression"), "compression")
    if set(profiles) != set(_PROFILE_METHODS):
        raise ValueError("compression must contain exactly baseline_bf16, int8_w8a8, and awq_w4a16.")
    for name in _PROFILE_METHODS:
        validate_compression_profile(name, get_compression_profile(config, name))
    validate_calibration_config(get_calibration_config(config), required=False)
    output = _mapping(config.get("output"), "output")
    _nonempty_string(output.get("compressed_model_root"), "output.compressed_model_root")


def validate_execution_config(config: Mapping[str, Any], profile_name: str) -> None:
    profile = get_compression_profile(config, profile_name)
    validate_compression_profile(profile_name, profile)
    if profile_name == "baseline_bf16":
        raise ValueError("BF16 baseline uses the original model; it is not compressed.")
    calibration = get_calibration_config(config)
    validate_calibration_config(calibration, required=True)
    if calibration["shuffle"] is not True:
        raise ValueError("Calibration shuffle must be true for both compression profiles.")
    if calibration["use_chat_template"] is not True:
        raise ValueError("Calibration use_chat_template must be true for both compression profiles.")


def resolve_compressed_model_path(
    config: Mapping[str, Any], profile_name: str, project_root: Path
) -> Path:
    get_compression_profile(config, profile_name)
    if profile_name == "baseline_bf16":
        raise ValueError("BF16 baseline has no copied compressed checkpoint.")
    model_id = _nonempty_string(get_model_config(config).get("model_id"), "model.model_id")
    model_slug = re.sub(r"[^a-z0-9]+", "-", model_id.split("/")[-1].lower()).strip("-")
    if not model_slug:
        raise ValueError("model.model_id cannot be converted to a directory name.")
    root = Path(_mapping(config.get("output"), "output")["compressed_model_root"])
    return (root if root.is_absolute() else Path(project_root) / root) / model_slug / profile_name


def build_experiment_metadata(
    config: Mapping[str, Any], profile_name: str
) -> dict[str, Any]:
    validate_execution_config(config, profile_name)
    model = get_model_config(config)
    profile = get_compression_profile(config, profile_name)
    calibration = get_calibration_config(config)
    return {
        "model_id": model["model_id"],
        "model_revision": model["revision"],
        "baseline_dtype": model["dtype"],
        "compression_profile": profile_name,
        "compression_method": profile["method"],
        "weight_bits": profile.get("weight_bits"),
        "activation_bits": profile.get("activation_bits"),
        "activation_dtype": profile.get("activation_dtype"),
        "activation_scheme": profile.get("activation_scheme"),
        "calibration_dataset": calibration.get("dataset"),
        "calibration_split": calibration.get("split"),
        "calibration_samples": calibration.get("num_samples"),
        "calibration_max_sequence_length": calibration.get("max_sequence_length"),
        "calibration_shuffle": calibration.get("shuffle"),
        "calibration_seed": calibration.get("seed"),
        "calibration_use_chat_template": calibration.get("use_chat_template"),
    }


def save_experiment_metadata(metadata: Mapping[str, Any], path: Path) -> None:
    """Serialize metadata only after a real compression run has succeeded."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(metadata), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_model_and_tokenizer(model_config: Mapping[str, Any]) -> tuple[Any, Any]:
    """Load the configured model only when a GPU pipeline is explicitly run."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = model_config["model_id"]
    options = {
        "revision": model_config["revision"],
        "trust_remote_code": model_config["trust_remote_code"],
    }
    tokenizer = AutoTokenizer.from_pretrained(model_id, **options)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=getattr(torch, model_config["dtype"]), **options
    )
    return model, tokenizer


def prepare_calibration_data(calibration: Mapping[str, Any], tokenizer: Any) -> Any:
    """Deterministically sample conversations for statistics, never training."""
    from datasets import load_dataset

    validate_calibration_config(calibration, required=True)
    if calibration["shuffle"] is not True or calibration["use_chat_template"] is not True:
        raise ValueError("Calibration requires shuffle and the model chat template.")
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("The configured tokenizer has no chat template.")
    dataset = load_dataset(calibration["dataset"], split=calibration["split"])
    sample_count = calibration["num_samples"]
    if len(dataset) < sample_count:
        raise ValueError(
            f"Calibration split contains {len(dataset)} rows; {sample_count} were requested."
        )
    dataset = dataset.shuffle(seed=calibration["seed"]).select(range(sample_count))
    max_length = calibration["max_sequence_length"]
    if "messages" not in dataset.column_names:
        raise ValueError("Calibration dataset must contain a 'messages' column for chat templating.")

    def tokenize(sample: Mapping[str, Any]) -> dict[str, Any]:
        content = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Calibration sample text must be non-empty.")
        return tokenizer(
            content,
            padding=False,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )

    return dataset.map(tokenize, remove_columns=dataset.column_names)
