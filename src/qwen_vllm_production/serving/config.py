"""Pure configuration and command construction; imports never start vLLM."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ServingConfig:
    model_path: str
    served_model_name: str = "qwen3-14b"
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    host: str = "0.0.0.0"
    port: int = 8000
    dtype: str = "bfloat16"
    seed: int = 42
    model_revision: str | None = None

    def __post_init__(self) -> None:
        if not self.model_path.strip():
            raise ValueError("MODEL_PATH / --model-path must not be empty.")
        if not self.served_model_name.strip() or not self.host.strip():
            raise ValueError("SERVED_MODEL_NAME and HOST must not be empty.")
        if self.max_model_len <= 0:
            raise ValueError("MAX_MODEL_LEN must be positive.")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("GPU_MEMORY_UTILIZATION must be in (0, 1].")
        if self.tensor_parallel_size != 1:
            raise ValueError("Phase 3A supports one GPU: TENSOR_PARALLEL_SIZE must be 1.")
        if not 1 <= self.port <= 65535:
            raise ValueError("PORT must be in [1, 65535].")
        if not self.dtype.strip():
            raise ValueError("DTYPE / --dtype must not be empty.")


def parse_config(argv: Sequence[str] | None = None, env: Mapping[str, str] | None = None) -> ServingConfig:
    env = os.environ if env is None else env
    parser = argparse.ArgumentParser(description="Launch the official vLLM OpenAI API server.")
    # Parse environment strings with the same argparse types used for CLI values.
    for flag, variable, default, kind in (
        ("model-path", "MODEL_PATH", "", str),
        ("served-model-name", "SERVED_MODEL_NAME", "qwen3-14b", str),
        ("max-model-len", "MAX_MODEL_LEN", "8192", int),
        ("gpu-memory-utilization", "GPU_MEMORY_UTILIZATION", "0.90", float),
        ("tensor-parallel-size", "TENSOR_PARALLEL_SIZE", "1", int),
        ("host", "HOST", "0.0.0.0", str),
        ("port", "PORT", "8000", int),
        ("dtype", "DTYPE", "bfloat16", str),
        ("seed", "SEED", "42", int),
        ("model-revision", "MODEL_REVISION", "", str),
    ):
        parser.add_argument("--" + flag, default=env.get(variable, default), type=kind)
    args = parser.parse_args(argv)
    args.model_revision = args.model_revision.strip() or None
    try:
        return ServingConfig(**vars(args))
    except ValueError as exc:
        parser.error(str(exc))


def build_command(config: ServingConfig) -> list[str]:
    """No variant branch or quantization override: vLLM reads checkpoint config."""
    command = [
        "vllm", "serve", config.model_path,
        "--served-model-name", config.served_model_name,
        "--max-model-len", str(config.max_model_len),
        "--gpu-memory-utilization", str(config.gpu_memory_utilization),
        "--tensor-parallel-size", str(config.tensor_parallel_size),
        "--host", config.host, "--port", str(config.port), "--dtype", config.dtype,
        "--seed", str(config.seed),
        "--no-enable-prefix-caching",
    ]
    if config.model_revision and config.model_revision.strip():
        command.extend(["--revision", config.model_revision.strip()])
    return command


def main() -> None:
    config = parse_config()
    # Replace the launcher with the official CLI so its PID and signals belong
    # to the server. Argument vector only, without shell interpolation.
    command = build_command(config)
    os.execvp(command[0], command)
