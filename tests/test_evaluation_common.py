"""Pure evaluation helpers; no model or dataset execution."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwen_vllm_production.evaluation.common import (
    MODEL_NAMES,
    result_path,
    save_json,
    validate_model_name,
)
from qwen_vllm_production.evaluation.sanity import (
    PROMPTS,
    build_parser,
    build_sanity_result,
    logits_are_finite,
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


def test_sanity_result_construction() -> None:
    result = build_sanity_result("int8_w8a8", "checkpoint", "cuda")
    assert result["evaluation"] == "sanity"
    assert result["load_success"] is False
    assert result["finite_logits"] is False
    assert result["prompts"] == list(PROMPTS)
    assert result["generation_settings"]["do_sample"] is False


def test_finite_logits_helper_uses_all_values(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeFinite:
        def __init__(self, values: list[bool]) -> None:
            self.values = values

        def all(self) -> "FakeFinite":
            return FakeFinite([all(self.values)])

        def item(self) -> bool:
            return self.values[0]

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(isfinite=lambda values: FakeFinite(values))
    )
    assert logits_are_finite([True, True]) is True
    assert logits_are_finite([True, False]) is False


def test_sanity_parser() -> None:
    args = build_parser().parse_args(
        ["--model-path", "checkpoint", "--model-name", "awq_w4a16", "--device", "cuda"]
    )
    assert args.model_name == "awq_w4a16"
    assert args.output_dir == Path("results/quality/sanity")
