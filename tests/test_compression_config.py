"""CPU-only checks of planned compression configuration and execution gates."""

from copy import deepcopy
from pathlib import Path

import pytest

from qwen_vllm_production.compression.common import (
    get_compression_profile,
    get_model_config,
    load_config,
    validate_config,
    validate_execution_config,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


@pytest.fixture
def config() -> dict:
    return load_config(CONFIG_PATH)


def test_unified_yaml_and_project(config: dict) -> None:
    assert config["project"]["name"] == "qwen-vllm-production"
    assert config["project"]["seed"] == 42


def test_model_and_profiles(config: dict) -> None:
    assert get_model_config(config)["model_id"] == "Qwen/Qwen3-14B"
    assert get_model_config(config)["dtype"] == "bfloat16"
    baseline = get_compression_profile(config, "baseline_bf16")
    int8 = get_compression_profile(config, "int8_w8a8")
    awq = get_compression_profile(config, "awq_w4a16")
    assert baseline["method"] == "none"
    assert baseline["requires_calibration"] is False
    assert int8["weight_bits"] == int8["activation_bits"] == 8
    assert int8["activation_scheme"] == "static"
    assert int8["requires_calibration"] is True
    assert awq["weight_bits"] == 4
    assert awq["requires_calibration"] is True
    assert "gptq" not in config["compression"]


def test_calibration_protocol(config: dict) -> None:
    calibration = config["calibration"]
    assert calibration["dataset"] == "HuggingFaceH4/ultrachat_200k"
    assert calibration["split"] == "train_sft"
    assert calibration["num_samples"] == 512
    assert calibration["max_sequence_length"] == 2048
    assert calibration["shuffle"] is True
    assert calibration["seed"] == 42
    assert calibration["use_chat_template"] is True


def test_unknown_profile_is_rejected(config: dict) -> None:
    with pytest.raises(ValueError, match="Unknown compression profile"):
        get_compression_profile(config, "missing")


def test_invalid_bit_width_is_rejected(config: dict) -> None:
    changed = deepcopy(config)
    changed["compression"]["awq_w4a16"]["weight_bits"] = 3
    with pytest.raises(ValueError, match="weight_bits"):
        validate_config(changed)


def test_missing_model_field_is_rejected(config: dict) -> None:
    changed = deepcopy(config)
    del changed["model"]["model_id"]
    with pytest.raises(ValueError, match="model.model_id"):
        validate_config(changed)


def test_awq_requires_complete_calibration_before_execution(config: dict) -> None:
    changed = deepcopy(config)
    changed["calibration"]["dataset"] = None
    with pytest.raises(ValueError, match="calibration.dataset"):
        validate_execution_config(changed, "awq_w4a16")


def test_int8_rejects_non_static_scheme(config: dict) -> None:
    changed = deepcopy(config)
    changed["compression"]["int8_w8a8"]["activation_scheme"] = "dynamic"
    with pytest.raises(ValueError, match="activation_scheme must be static"):
        validate_execution_config(changed, "int8_w8a8")


def test_static_int8_requires_calibration(config: dict) -> None:
    changed = deepcopy(config)
    changed["compression"]["int8_w8a8"]["requires_calibration"] = False
    with pytest.raises(ValueError, match="Static W8A8 requires calibration"):
        validate_execution_config(changed, "int8_w8a8")


def test_baseline_does_not_require_calibration(config: dict) -> None:
    changed = deepcopy(config)
    changed["calibration"] = {}
    with pytest.raises(ValueError, match="BF16 baseline uses the original model"):
        validate_execution_config(changed, "baseline_bf16")
