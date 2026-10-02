"""W4A16 AWQ compression entrypoint for a future Linux GPU run."""

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

PROFILE_NAME = "awq_w4a16"


def build_awq_recipe(profile: Mapping[str, Any]) -> list[Any]:
    """Build the documented AWQ transform followed by W4A16 quantization."""
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier

    # TODO: Verify default AWQ layer mappings for Qwen3-14B on the Colab stack.
    scheme = f"W{profile['weight_bits']}A16"
    return [
        AWQModifier(),
        QuantizationModifier(targets="Linear", scheme=scheme, ignore=["lm_head"]),
    ]


def run(config_path: Path) -> Path:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    validate_execution_config(config, PROFILE_NAME)
    profile = get_compression_profile(config, PROFILE_NAME)
    recipe = build_awq_recipe(profile)
    model, tokenizer = load_model_and_tokenizer(get_model_config(config))
    calibration = get_calibration_config(config)
    dataset = prepare_calibration_data(calibration, tokenizer, seed=config["project"]["seed"])
    output_path = resolve_compressed_model_path(config, PROFILE_NAME, config_path.parent.parent)

    from llmcompressor import oneshot

    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=calibration["max_sequence_length"],
        num_calibration_samples=calibration["num_samples"],
        output_dir=str(output_path),
        save_compressed=True,
    )
    save_experiment_metadata(
        build_experiment_metadata(config, PROFILE_NAME), output_path / "experiment_metadata.json"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress the configured model with AWQ.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
