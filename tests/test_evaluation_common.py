"""Mock-only checks of shared vLLM settings and offline sanity output."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_vllm_production.evaluation.common import (
    MODEL_NAMES,
    local_checkpoint_metadata,
    result_path,
    save_json,
    validate_model_name,
    vllm_engine_config,
    vllm_harness_model_args,
)
from qwen_vllm_production.evaluation.sanity import (
    PROMPTS,
    build_parser,
    build_sanity_result,
    evaluate,
    extract_generated_samples,
)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_names(name: str) -> None:
    assert validate_model_name(name) == name


def test_invalid_model_name() -> None:
    with pytest.raises(ValueError, match="model_name"):
        validate_model_name("other")


def test_output_path_and_json(tmp_path: Path) -> None:
    path = result_path(tmp_path / "sanity", "bf16")
    assert path == tmp_path / "sanity" / "bf16.json"
    save_json({"path": tmp_path, "ok": True}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "path": str(tmp_path), "ok": True,
    }


def test_json_serializes_callable_set_and_unknown_object(tmp_path: Path) -> None:
    def example_function() -> None:
        pass

    class Unknown:
        def __str__(self) -> str:
            return "unknown-object"

    path = tmp_path / "raw_results.json"
    save_json({
        "raw_results": {
            "metric": 1.0,
            "task_function": example_function,
            "labels": {"test", "validation"},
            "unknown": Unknown(),
        },
    }, path)
    raw_results = json.loads(path.read_text(encoding="utf-8"))["raw_results"]
    assert raw_results["metric"] == 1.0
    assert isinstance(raw_results["task_function"], str)
    assert raw_results["task_function"].endswith(".example_function")
    assert set(raw_results["labels"]) == {"test", "validation"}
    assert raw_results["unknown"] == "unknown-object"


def test_json_preserves_existing_object_conversions(tmp_path: Path) -> None:
    class WithDict:
        def to_dict(self) -> dict[str, int]:
            return {"value": 1}

    class WithList:
        def tolist(self) -> list[int]:
            return [1, 2]

    class WithItem:
        def item(self) -> int:
            return 3

    path = tmp_path / "conversions.json"
    save_json({
        "path": tmp_path,
        "dict": WithDict(),
        "list": WithList(),
        "scalar": WithItem(),
    }, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "path": str(tmp_path),
        "dict": {"value": 1},
        "list": [1, 2],
        "scalar": 3,
    }


def test_shared_engine_config_and_harness_adapter() -> None:
    config = vllm_engine_config("Qwen/Qwen3-14B", max_model_len=4096, seed=7)
    assert config == {
        "model": "Qwen/Qwen3-14B",
        "tensor_parallel_size": 1,
        "dtype": "auto",
        "gpu_memory_utilization": 0.9,
        "trust_remote_code": False,
        "seed": 7,
        "max_model_len": 4096,
    }
    assert vllm_harness_model_args(config) == {
        "pretrained": "Qwen/Qwen3-14B",
        **{key: value for key, value in config.items() if key != "model"},
    }
    assert "quantization" not in config


@pytest.mark.parametrize("kwargs", [
    {"tensor_parallel_size": 0},
    {"gpu_memory_utilization": 1.1},
    {"max_model_len": 0},
    {"seed": True},
])
def test_invalid_engine_config(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        vllm_engine_config("checkpoint", **kwargs)


def test_local_quantization_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({
        "_name_or_path": "Qwen/Qwen3-14B",
        "torch_dtype": "bfloat16",
        "quantization_config": {"quant_method": "compressed-tensors"},
    }), encoding="utf-8")
    metadata = local_checkpoint_metadata(checkpoint)
    assert metadata["quantization_method"] == "compressed-tensors"
    assert metadata["checkpoint_dtype"] == "bfloat16"


def test_sanity_result_construction() -> None:
    result = build_sanity_result("int8_w8a8", "checkpoint", "cuda")
    assert result["evaluation"] == "sanity"
    assert result["backend"] == "vllm"
    assert result["engine_load_success"] is False
    assert result["generation_success"] is False
    assert result["prompts"] == list(PROMPTS)
    assert result["generation_settings"] == {"temperature": 0.0, "max_tokens": 8}
    assert result["dtype_configuration"] == "auto"


def test_generated_sample_extraction() -> None:
    outputs = [
        SimpleNamespace(prompt=prompt, outputs=[SimpleNamespace(text=f" answer {index}")])
        for index, prompt in enumerate(PROMPTS)
    ]
    samples = extract_generated_samples(outputs, PROMPTS)
    assert [sample["prompt"] for sample in samples] == list(PROMPTS)
    assert samples[0]["generated_text"] == " answer 0"
    outputs[1].outputs[0].text = "  "
    with pytest.raises(ValueError, match="empty text"):
        extract_generated_samples(outputs, PROMPTS)


def test_mocked_sanity_inference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, object] = {}

    class FakeSamplingParams:
        def __init__(self, **kwargs: object) -> None:
            calls["sampling"] = kwargs

    class FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            calls["engine"] = kwargs
            self.llm_engine = SimpleNamespace(model_config=SimpleNamespace(
                model="checkpoint", dtype="bfloat16", quantization="compressed-tensors",
                max_model_len=4096,
            ))

        def generate(self, prompts: list[str], sampling_params: object) -> list[object]:
            calls["prompts"] = prompts
            assert isinstance(sampling_params, FakeSamplingParams)
            return [SimpleNamespace(
                prompt=prompt, outputs=[SimpleNamespace(text=" continuation")]
            ) for prompt in prompts]

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        LLM=FakeLLM, SamplingParams=FakeSamplingParams,
    ))
    result = evaluate(
        "checkpoint", "awq_w4a16", "cuda", tmp_path,
        max_model_len=4096, seed=7,
    )
    assert calls["engine"] == vllm_engine_config("checkpoint", max_model_len=4096, seed=7)
    assert calls["sampling"] == {"temperature": 0.0, "max_tokens": 8}
    assert calls["prompts"] == list(PROMPTS)
    assert result["engine_load_success"] is True
    assert result["generation_success"] is True
    assert result["runtime_quantization"] == "compressed-tensors"
    assert len(result["generated_samples"]) == len(PROMPTS)
    assert json.loads((tmp_path / "awq_w4a16.json").read_text(encoding="utf-8"))["backend"] == "vllm"


def test_sanity_parser() -> None:
    args = build_parser().parse_args([
        "--model-path", "checkpoint", "--model-name", "awq_w4a16",
        "--device", "cuda", "--tensor-parallel-size", "1",
        "--gpu-memory-utilization", "0.85", "--max-model-len", "4096", "--seed", "7",
    ])
    assert args.model_name == "awq_w4a16"
    assert args.tensor_parallel_size == 1
    assert args.gpu_memory_utilization == 0.85
    assert args.max_model_len == 4096
    assert args.seed == 7
    assert args.output_dir == Path("results/quality/sanity")
