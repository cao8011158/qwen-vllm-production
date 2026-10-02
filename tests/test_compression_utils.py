"""CPU-only utility checks; no model, network, or GPU access."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from compression.common import (
    build_experiment_metadata,
    get_calibration_config,
    get_compression_profile,
    get_model_config,
    load_config,
    resolve_compressed_model_path,
    save_experiment_metadata,
    validate_config,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


@pytest.fixture
def config() -> dict:
    return load_config(CONFIG_PATH)


def test_yaml_parsing_and_extraction(config: dict) -> None:
    assert get_model_config(config)["revision"] == "main"
    assert get_compression_profile(config, "int8_w8a8")["activation_scheme"] is None
    assert get_calibration_config(config)["dataset"] is None


def test_path_resolution_is_portable(config: dict, tmp_path: Path) -> None:
    expected = tmp_path / "outputs" / "compressed" / "qwen3-14b" / "awq_w4a16"
    assert resolve_compressed_model_path(config, "awq_w4a16", tmp_path) == expected
    assert not expected.exists()


def test_baseline_has_no_checkpoint_copy(config: dict, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no copied compressed checkpoint"):
        resolve_compressed_model_path(config, "baseline_bf16", tmp_path)


def test_metadata_construction_and_json(config: dict, tmp_path: Path) -> None:
    changed = deepcopy(config)
    changed["calibration"].update(
        dataset="example/calibration", split="train", num_samples=8,
        max_sequence_length=128,
    )
    metadata = build_experiment_metadata(changed, "awq_w4a16")
    assert metadata["model_id"] == changed["model"]["model_id"]
    assert metadata["compression_method"] == "awq"
    assert metadata["calibration_samples"] == 8
    path = tmp_path / "run" / "experiment_metadata.json"
    save_experiment_metadata(metadata, path)
    assert json.loads(path.read_text(encoding="utf-8")) == metadata


def test_invalid_configuration_is_rejected(config: dict) -> None:
    changed = deepcopy(config)
    changed["calibration"]["num_samples"] = -1
    with pytest.raises(ValueError, match="calibration.num_samples"):
        validate_config(changed)


def test_invalid_yaml_root_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a mapping"):
        load_config(path)
