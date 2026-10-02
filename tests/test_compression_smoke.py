"""Opt-in Linux GPU pipeline smoke test; never a formal quality result."""

import os
from pathlib import Path

import pytest

from compression.common import get_calibration_config, load_config, validate_calibration_config


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.integration,
    pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1", reason="Set RUN_GPU_TESTS=1 explicitly."),
]

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def test_awq_load_calibrate_compress_save_reload(tmp_path: Path) -> None:
    model_id = os.getenv("SMOKE_TEST_MODEL_ID")
    if not model_id:
        pytest.skip("Set SMOKE_TEST_MODEL_ID to a small compatible Qwen model.")
    config = load_config(CONFIG_PATH)
    try:
        validate_calibration_config(get_calibration_config(config), required=True)
    except ValueError as exc:
        pytest.skip(f"Configure calibration parameters before GPU smoke testing: {exc}")

    # The override exists only in this temporary smoke configuration.
    config["model"]["model_id"] = model_id
    config["output"]["compressed_model_root"] = str(tmp_path / "compressed")
    smoke_config = tmp_path / "configs" / "config.yaml"
    smoke_config.parent.mkdir(parents=True)
    import yaml

    smoke_config.write_text(yaml.safe_dump(config), encoding="utf-8")
    from compression.compress_awq import run

    output_path = run(smoke_config)
    assert (output_path / "experiment_metadata.json").is_file()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    AutoTokenizer.from_pretrained(output_path, local_files_only=True)
    AutoModelForCausalLM.from_pretrained(output_path, local_files_only=True)
