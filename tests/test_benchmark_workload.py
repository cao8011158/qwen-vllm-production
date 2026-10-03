"""Synthetic tokenizer only: no tokenizer or dataset download."""

import sys
from types import SimpleNamespace

import pytest

from qwen_vllm_production.benchmark.workload import build_workload, load_local_tokenizer, validate_workload


class FakeTokenizer:
    def encode(self, text: str, **kwargs) -> list[str]:
        return text.split()

    def apply_chat_template(self, messages: list[dict], **kwargs) -> str:
        assert kwargs["enable_thinking"] is False
        return "system " + messages[0]["content"] + " user " + messages[1]["content"] + " assistant"


def test_deterministic_token_sized_workload() -> None:
    tokenizer = FakeTokenizer()
    kwargs = dict(count=3, seed=42, target_input_tokens=1000, target_output_tokens=256)
    first = build_workload(tokenizer, **kwargs)
    second = build_workload(tokenizer, **kwargs)
    assert first == second
    assert all(abs(record["prompt_tokens"] - 1000) <= 16 for record in first["prompts"])
    assert len({record["sha256"] for record in first["prompts"]}) == 3
    validate_workload(first, tokenizer, **kwargs)
    assert first["workload_sha256"] != build_workload(tokenizer, **{**kwargs, "seed": 43})["workload_sha256"]
    first["prompts"][0]["text"] += "changed"
    with pytest.raises(ValueError, match="hash"):
        validate_workload(first, tokenizer, **kwargs)


def test_tokenizer_loader_forbids_download(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_load(path: str, **kwargs):
        calls.append((path, kwargs))
        return FakeTokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=fake_load),
    ))
    load_local_tokenizer("/local/tokenizer")
    assert calls == [("/local/tokenizer", {"local_files_only": True, "trust_remote_code": False})]
