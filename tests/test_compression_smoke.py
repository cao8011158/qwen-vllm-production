"""Opt-in GPU smoke tests for the compression pipelines.

These tests are intended only to verify that the engineering pipeline can:

load -> calibrate -> compress -> save -> reload

They are NOT formal quality or performance experiments.

Formal experiment configuration remains unchanged:
- Qwen/Qwen3-14B
- 512 calibration samples
- 2048 max calibration tokens

Smoke tests temporarily override these values with a much smaller workload.
"""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Callable

import pytest
import yaml

from qwen_vllm_production.compression.common import load_config


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_GPU_TESTS") != "1",
        reason="Set RUN_GPU_TESTS=1 explicitly to enable GPU smoke tests.",
    ),
]


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"

SMOKE_CALIBRATION_SAMPLES = 8
SMOKE_MAX_SEQUENCE_LENGTH = 256


def _require_gpu() -> None:
    """Fail early if GPU smoke tests are enabled without a CUDA GPU."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is required for compression smoke tests.")


def _create_smoke_config(tmp_path: Path) -> Path:
    """Create a temporary lightweight configuration for GPU smoke testing."""
    model_id = os.getenv("SMOKE_TEST_MODEL_ID")

    if not model_id:
        pytest.skip(
            "Set SMOKE_TEST_MODEL_ID to a small compatible model, "
            "for example Qwen/Qwen3-0.6B."
        )

    config = deepcopy(load_config(CONFIG_PATH))

    # Override only the values needed to keep the smoke test lightweight.
    # The repository's formal config.yaml remains unchanged.
    config["model"]["model_id"] = model_id

    config["calibration"]["num_samples"] = SMOKE_CALIBRATION_SAMPLES
    config["calibration"]["max_sequence_length"] = SMOKE_MAX_SEQUENCE_LENGTH

    config["output"]["compressed_model_root"] = str(
        tmp_path / "compressed"
    )

    smoke_config_path = tmp_path / "configs" / "config.yaml"
    smoke_config_path.parent.mkdir(parents=True, exist_ok=True)

    smoke_config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    return smoke_config_path


def _verify_saved_model(output_path: Path) -> None:
    """Verify that the compressed checkpoint can be reloaded locally."""
    assert output_path.is_dir()
    assert (output_path / "experiment_metadata.json").is_file()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    AutoTokenizer.from_pretrained(
        output_path,
        local_files_only=True,
    )

    AutoModelForCausalLM.from_pretrained(
        output_path,
        local_files_only=True,
    )


def test_awq_load_calibrate_compress_save_reload(tmp_path: Path) -> None:
    """Smoke-test the W4A16 AWQ compression pipeline."""
    _require_gpu()

    smoke_config = _create_smoke_config(tmp_path)

    from qwen_vllm_production.compression.compress_awq import run

    output_path = run(smoke_config)

    _verify_saved_model(output_path)


def test_int8_load_calibrate_compress_save_reload(tmp_path: Path) -> None:
    """Smoke-test the static W8A8 INT8 compression pipeline."""
    _require_gpu()

    smoke_config = _create_smoke_config(tmp_path)

    from qwen_vllm_production.compression.compress_int8 import run

    output_path = run(smoke_config)

    _verify_saved_model(output_path)