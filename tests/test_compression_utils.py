"""CPU-only utility checks; no model, network, or GPU access."""

import json
import sys
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from compression.common import (
    build_experiment_metadata,
    get_calibration_config,
    get_compression_profile,
    get_model_config,
    load_config,
    prepare_calibration_data,
    resolve_compressed_model_path,
    save_experiment_metadata,
    validate_calibration_config,
    validate_config,
)


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


@pytest.fixture
def config() -> dict:
    return load_config(CONFIG_PATH)


def test_yaml_parsing_and_extraction(config: dict) -> None:
    assert get_model_config(config)["revision"] == "main"
    assert get_compression_profile(config, "int8_w8a8")["activation_scheme"] == "static"
    assert get_calibration_config(config)["dataset"] == "HuggingFaceH4/ultrachat_200k"


def test_valid_calibration_config(config: dict) -> None:
    validate_calibration_config(get_calibration_config(config))


def test_calibration_shuffles_full_split_before_selection_and_uses_chat_template(
    config: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[object] = []
    calibration = deepcopy(config["calibration"])
    calibration["num_samples"] = 2
    calibration["max_sequence_length"] = 2

    class FakeDataset:
        column_names = ["messages"]

        def __init__(self) -> None:
            self.rows = [
                {"messages": [{"content": "first one two"}]},
                {"messages": [{"content": "second one two"}]},
                {"messages": [{"content": "third one two"}]},
            ]

        def __len__(self) -> int:
            return len(self.rows)

        def shuffle(self, *, seed: int) -> "FakeDataset":
            events.append(("shuffle", seed, len(self.rows)))
            self.rows = [self.rows[index] for index in (2, 0, 1)]
            return self

        def select(self, indices: range) -> "FakeDataset":
            events.append(("select", list(indices)))
            self.rows = [self.rows[index] for index in indices]
            return self

        def map(
            self, function: Callable[[dict], dict], *, remove_columns: list[str]
        ) -> list[dict]:
            assert remove_columns == ["messages"]
            return [function(row) for row in self.rows]

    class FakeTokenizer:
        chat_template = "model supplied template"

        def apply_chat_template(self, messages: list[dict], *, tokenize: bool) -> str:
            assert tokenize is False
            events.append("chat_template")
            return messages[0]["content"]

        def __call__(self, text: str, **kwargs: object) -> dict:
            assert kwargs == {
                "padding": False,
                "truncation": True,
                "max_length": 2,
                "add_special_tokens": False,
            }
            return {"input_ids": text.split()[:2]}

    def fake_load_dataset(dataset: str, *, split: str) -> FakeDataset:
        assert (dataset, split) == (calibration["dataset"], calibration["split"])
        return FakeDataset()

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset))
    selected = prepare_calibration_data(calibration, FakeTokenizer())
    assert events[:2] == [("shuffle", 42, 3), ("select", [0, 1])]
    assert selected == [
        {"input_ids": ["third", "one"]},
        {"input_ids": ["first", "one"]},
    ]


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
    assert metadata["calibration_shuffle"] is True
    assert metadata["calibration_seed"] == 42
    assert metadata["calibration_use_chat_template"] is True
    path = tmp_path / "run" / "experiment_metadata.json"
    save_experiment_metadata(metadata, path)
    assert json.loads(path.read_text(encoding="utf-8")) == metadata


def test_invalid_configuration_is_rejected(config: dict) -> None:
    changed = deepcopy(config)
    changed["calibration"]["num_samples"] = -1
    with pytest.raises(ValueError, match="calibration.num_samples"):
        validate_config(changed)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("num_samples", 0),
        ("max_sequence_length", -1),
        ("seed", True),
        ("shuffle", "yes"),
        ("use_chat_template", "yes"),
    ],
)
def test_invalid_calibration_field_is_rejected(
    config: dict, field: str, value: object
) -> None:
    changed = deepcopy(config["calibration"])
    changed[field] = value
    with pytest.raises(ValueError, match=f"calibration.{field}"):
        validate_calibration_config(changed)


def test_invalid_yaml_root_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- not\n- a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a mapping"):
        load_config(path)
