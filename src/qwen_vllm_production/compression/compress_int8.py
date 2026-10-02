"""Static W8A8 INT8 compression pipeline for a future Linux GPU run."""

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
    """Describe per-channel INT8 weights and static per-tensor INT8 inputs."""
    if profile.get("activation_scheme") != "static":
        raise ValueError("int8_w8a8.activation_scheme must be static.")
    from compressed_tensors.quantization import QuantizationScheme
    from compressed_tensors.quantization.quant_args import (
        QuantizationArgs,
        QuantizationStrategy,
        QuantizationType,
    )
    from llmcompressor.modifiers.quantization import QuantizationModifier

    # TODO: Confirm this explicit static scheme and Qwen3/vLLM runtime behavior
    # against the installed Colab versions before a formal compression run.
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=profile["weight_bits"],
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.CHANNEL,
            symmetric=True,
            dynamic=False,
        ),
        input_activations=QuantizationArgs(
            num_bits=profile["activation_bits"],
            type=QuantizationType.INT,
            strategy=QuantizationStrategy.TENSOR,
            symmetric=True,
            dynamic=False,
        ),
    )
    return QuantizationModifier(config_groups={"group_0": scheme}, ignore=["lm_head"])


def run(config_path: Path) -> Path:
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    validate_execution_config(config, PROFILE_NAME)
    profile = get_compression_profile(config, PROFILE_NAME)
    recipe = build_int8_recipe(profile)
    model, tokenizer = load_model_and_tokenizer(get_model_config(config))
    calibration = get_calibration_config(config)
    dataset = prepare_calibration_data(calibration, tokenizer)
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
    parser = argparse.ArgumentParser(description="Compress the configured model with INT8.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args()
    print(run(args.config))


if __name__ == "__main__":
    main()
