"""W4A16 AWQ compression entrypoint for a Linux GPU run."""

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
    """
    Build the AWQ transform followed by W4A16 quantization.

    AWQ caches calibration inputs while searching for weight scaling factors.
    For dense models such as Qwen3-14B, those cached arguments are not
    offloaded by default, which can cause large GPU-memory spikes.

    Explicitly offload the AWQ cache to CPU to reduce VRAM usage.
    """
    import torch

    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier

    scheme = f"W{profile['weight_bits']}A16"

    return [
        AWQModifier(
            offload_device=torch.device("cpu"),
        ),
        QuantizationModifier(
            targets="Linear",
            scheme=scheme,
            ignore=["lm_head"],
        ),
    ]


def run(config_path: Path) -> Path:
    """Run W4A16 AWQ compression using the configured calibration dataset."""

    config_path = Path(config_path).resolve()

    config = load_config(config_path)
    validate_execution_config(config, PROFILE_NAME)

    profile = get_compression_profile(config, PROFILE_NAME)
    calibration = get_calibration_config(config)

    recipe = build_awq_recipe(profile)

    # common.py loads the model on CPU.
    # llmcompressor will sequentially onload the required modules to the GPU.
    model, tokenizer = load_model_and_tokenizer(
        get_model_config(config)
    )

    dataset = prepare_calibration_data(
        calibration,
        tokenizer,
    )

    output_path = resolve_compressed_model_path(
        config,
        PROFILE_NAME,
        config_path.parent.parent,
    )

    from llmcompressor import oneshot

    oneshot(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        recipe=recipe,

        # Calibration configuration
        max_seq_length=calibration["max_sequence_length"],
        num_calibration_samples=calibration["num_samples"],
        batch_size=1,

        # Reduce peak GPU-memory usage.
        #
        # Instead of using an entire DecoderLayer as the sequential
        # subgraph, operate on smaller Linear modules.
        sequential_targets=["Linear"],

        # Explicit for reproducibility. This is normally already the
        # default, but makes our memory-management policy clear.
        sequential_offload_device="cpu",

        output_dir=str(output_path),
        save_compressed=True,
    )

    save_experiment_metadata(
        build_experiment_metadata(config, PROFILE_NAME),
        output_path / "experiment_metadata.json",
    )

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compress the configured model with W4A16 AWQ."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
    )

    args = parser.parse_args()

    output_path = run(args.config)
    print(output_path)


if __name__ == "__main__":
    main()