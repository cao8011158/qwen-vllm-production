"""W8A8 INT8 pipeline, gated until the activation design is chosen."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .common import (
    build_experiment_metadata,
    get_calibration_config,
    get_compression_profile,
    get_model_config,
    load_config,
    load_model_and_tokenizer,
    prepare_calibration_data,
    resolve_compressed_model_path,
    save_experiment_metadata,
    validate_execution_config,
)

PROFILE_NAME = "int8_w8a8"


def build_int8_recipe(profile: Mapping[str, Any]) -> Any:
    """Refuse to invent a static/dynamic recipe for the selected scheme."""
    # TODO: After choosing static or dynamic, verify an INT8-only recipe against
    # the installed LLM Compressor API and Qwen3/vLLM compatibility. The official
    # W8A8 example uses a weight-quantization component excluded by this project.
    raise NotImplementedError(
        f"No verified INT8 recipe for activation_scheme={profile['activation_scheme']!r}; "
        "confirm the LLM Compressor API in Colab before compression."
    )


def run(config_path: Path) -> Path:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    validate_execution_config(config, PROFILE_NAME)
    profile = get_compression_profile(config, PROFILE_NAME)
    recipe = build_int8_recipe(profile)  # Fail before any model or dataset download.
    model, tokenizer = load_model_and_tokenizer(get_model_config(config))
    calibration = get_calibration_config(config)
    dataset = (
        prepare_calibration_data(calibration, tokenizer, seed=config["project"]["seed"])
        if profile["requires_calibration"]
        else None
    )
    output_path = resolve_compressed_model_path(config, PROFILE_NAME, config_path.parent.parent)

    from llmcompressor import oneshot

    options: dict[str, Any] = {}
    if dataset is not None:
        options = {
            "dataset": dataset,
            "max_seq_length": calibration["max_sequence_length"],
            "num_calibration_samples": calibration["num_samples"],
        }
    oneshot(
        model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        output_dir=str(output_path),
        save_compressed=True,
        **options,
    )
    save_experiment_metadata(
        build_experiment_metadata(config, PROFILE_NAME), output_path / "experiment_metadata.json"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress the configured model with INT8.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
