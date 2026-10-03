"""Serving argument construction only; never invoke the launcher subprocess."""

import pytest

from qwen_vllm_production.serving.config import ServingConfig, build_command, parse_config


def test_defaults_and_environment() -> None:
    config = parse_config([], {"MODEL_PATH": "Qwen/Qwen3-14B"})
    assert config == ServingConfig("Qwen/Qwen3-14B")
    assert config.max_model_len == 8192
    assert config.gpu_memory_utilization == 0.90
    assert config.tensor_parallel_size == 1
    assert config.host == "0.0.0.0"
    assert config.port == 8000
    custom = parse_config([], {
        "MODEL_PATH": "/local/awq", "SERVED_MODEL_NAME": "test-model",
        "MAX_MODEL_LEN": "4096", "GPU_MEMORY_UTILIZATION": "0.85",
        "TENSOR_PARALLEL_SIZE": "1", "HOST": "127.0.0.1", "PORT": "8080",
    })
    assert (custom.max_model_len, custom.gpu_memory_utilization, custom.host, custom.port) == (4096, 0.85, "127.0.0.1", 8080)


def test_cli_overrides_environment() -> None:
    config = parse_config(["--model-path", "/local/awq", "--port", "8001"], {
        "MODEL_PATH": "Qwen/Qwen3-14B", "PORT": "8000",
    })
    assert config.model_path == "/local/awq"
    assert config.port == 8001
    baseline = build_command(ServingConfig("Qwen/Qwen3-14B"))
    candidate = build_command(ServingConfig("/local/awq"))
    assert baseline[:2] == ["vllm", "serve"]
    assert baseline[:2] + baseline[3:] == candidate[:2] + candidate[3:]
    assert "--quantization" not in baseline
    assert "--no-enable-prefix-caching" in baseline


@pytest.mark.parametrize("env", [
    {}, {"MODEL_PATH": " "}, {"MODEL_PATH": "model", "MAX_MODEL_LEN": "0"},
    {"MODEL_PATH": "model", "GPU_MEMORY_UTILIZATION": "1.1"},
    {"MODEL_PATH": "model", "TENSOR_PARALLEL_SIZE": "2"},
    {"MODEL_PATH": "model", "HOST": ""}, {"MODEL_PATH": "model", "PORT": "70000"},
    {"MODEL_PATH": "model", "PORT": "not-a-number"},
])
def test_invalid_configuration(env: dict) -> None:
    with pytest.raises(SystemExit):
        parse_config([], env)
