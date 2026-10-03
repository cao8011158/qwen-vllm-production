"""Serving argument construction only; never invoke the launcher subprocess."""

from pathlib import Path

import pytest
import yaml

from qwen_vllm_production.serving.config import ServingConfig, build_command, parse_config


def test_defaults_and_environment() -> None:
    config = parse_config([], {"MODEL_PATH": "Qwen/Qwen3-14B"})
    assert config == ServingConfig("Qwen/Qwen3-14B")
    assert config.max_model_len == 8192
    assert config.gpu_memory_utilization == 0.90
    assert config.tensor_parallel_size == 1
    assert config.host == "0.0.0.0"
    assert config.port == 8000
    assert config.dtype == "bfloat16"
    assert config.seed == 42
    assert config.model_revision is None
    custom = parse_config([], {
        "MODEL_PATH": "/local/awq", "SERVED_MODEL_NAME": "test-model",
        "MAX_MODEL_LEN": "4096", "GPU_MEMORY_UTILIZATION": "0.85",
        "TENSOR_PARALLEL_SIZE": "1", "HOST": "127.0.0.1", "PORT": "8080",
        "DTYPE": "float16", "SEED": "7", "MODEL_REVISION": "revision-from-env",
    })
    assert (custom.max_model_len, custom.gpu_memory_utilization, custom.host, custom.port) == (4096, 0.85, "127.0.0.1", 8080)
    assert (custom.dtype, custom.seed, custom.model_revision) == ("float16", 7, "revision-from-env")


def test_cli_overrides_environment() -> None:
    config = parse_config([
        "--model-path", "/local/awq", "--port", "8001",
        "--dtype", "bfloat16", "--seed", "42", "--model-revision", "revision-from-cli",
    ], {
        "MODEL_PATH": "Qwen/Qwen3-14B", "PORT": "8000",
        "DTYPE": "float16", "SEED": "7", "MODEL_REVISION": "revision-from-env",
    })
    assert config.model_path == "/local/awq"
    assert config.port == 8001
    assert (config.dtype, config.seed, config.model_revision) == ("bfloat16", 42, "revision-from-cli")
    baseline = build_command(ServingConfig("Qwen/Qwen3-14B"))
    candidate = build_command(ServingConfig("/local/awq"))
    assert baseline[:2] == ["vllm", "serve"]
    assert baseline[:2] + baseline[3:] == candidate[:2] + candidate[3:]
    assert "--quantization" not in baseline
    assert "--no-enable-prefix-caching" in baseline
    assert baseline[baseline.index("--dtype") + 1] == "bfloat16"
    assert baseline[baseline.index("--seed") + 1] == "42"
    assert "--revision" not in baseline
    revision_command = build_command(config)
    assert revision_command[revision_command.index("--revision") + 1] == "revision-from-cli"


@pytest.mark.parametrize("revision", ["", "   "])
def test_empty_local_checkpoint_revision_is_omitted(revision: str) -> None:
    config = parse_config([], {"MODEL_PATH": "/local/awq", "MODEL_REVISION": revision})
    assert config.model_revision is None
    assert "--revision" not in build_command(config)


def test_project_config_matches_formal_protocol() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["slo"]["quality"]["recovery_minimum"] == 0.97
    assert "structured_output" not in config["evaluation"]
    assert "structured_output" not in config["slo"]
    assert "long_context" not in config["workloads"]
    assert config["workloads"]["interactive_chat"] == {
        "input_tokens": 1000, "output_tokens": 256, "max_context_tokens": 8192,
    }
    assert config["serving"]["gpu_memory_utilization"] == 0.90
    assert config["serving"]["tensor_parallel_size"] == 1


@pytest.mark.parametrize("env", [
    {}, {"MODEL_PATH": " "}, {"MODEL_PATH": "model", "MAX_MODEL_LEN": "0"},
    {"MODEL_PATH": "model", "GPU_MEMORY_UTILIZATION": "1.1"},
    {"MODEL_PATH": "model", "TENSOR_PARALLEL_SIZE": "2"},
    {"MODEL_PATH": "model", "HOST": ""}, {"MODEL_PATH": "model", "PORT": "70000"},
    {"MODEL_PATH": "model", "PORT": "not-a-number"},
    {"MODEL_PATH": "model", "DTYPE": " "},
    {"MODEL_PATH": "model", "SEED": "not-a-number"},
])
def test_invalid_configuration(env: dict) -> None:
    with pytest.raises(SystemExit):
        parse_config([], env)
