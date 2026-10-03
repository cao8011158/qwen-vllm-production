"""Async scheduling with fake requests; no server or GPU."""

import asyncio
from collections import Counter
from pathlib import Path

import pytest

from qwen_vllm_production.benchmark import runner
from qwen_vllm_production.benchmark.metrics import RequestResult
from qwen_vllm_production.benchmark.runner import build_parser, run_point
from qwen_vllm_production.benchmark.sweep import closed_loop
from qwen_vllm_production.benchmark.workload import Prompt


def prompts(count: int) -> list[Prompt]:
    return [Prompt(index, f"prompt {index}", 1000, str(index)) for index in range(count)]


def test_closed_loop_limits_outstanding_and_continues_after_failure() -> None:
    async def invoke():
        outstanding, peak = 0, 0
        workers = Counter()

        async def send(prompt, request_id, worker_id, concurrency):
            nonlocal outstanding, peak
            outstanding += 1
            peak = max(peak, outstanding)
            workers[worker_id] += 1
            await asyncio.sleep(0)
            outstanding -= 1
            if request_id == 5:
                raise RuntimeError("fake request failure")
            return RequestResult(request_id, "model", concurrency, worker_id, "start", 1000)

        records = await closed_loop(prompts(17), 4, send, model="model")
        assert peak == 4
        assert outstanding == 0
        assert all(count > 1 for count in workers.values())
        assert len(records) == 17
        assert [record.request_id for record in records] == list(range(17))
        assert records[5].error_message == "fake request failure"

    asyncio.run(invoke())


def test_warmup_excluded_from_measurements() -> None:
    class FakeClient:
        model = "model"

        async def send(self, prompt, request_id, worker_id, concurrency):
            return RequestResult(
                request_id, self.model, concurrency, worker_id, "start", 1000,
                success=True, output_tokens=256,
                ttft_ms=100, tpot_ms_per_token=20, e2e_ms=5200,
            )

    ticks = iter([10.0, 12.0])
    # A UTC interval of 10 seconds must not replace the monotonic duration of 2.
    utc_ticks = iter(["2026-01-01T00:00:00+00:00", "2026-01-01T00:00:10+00:00"])
    records, point = asyncio.run(run_point(
        FakeClient(), prompts(8), concurrency=2, warmup_requests=3,
        metadata={}, clock=lambda: next(ticks),
        utc_clock=lambda: next(utc_ticks),
    ))
    assert len(records) == point["num_requests"] == 5
    assert point["warmup_successful"] == 3
    assert point["benchmark_duration_seconds"] == 2.0
    assert point["throughput_requests_per_second"] == 2.5
    assert point["measurement_start_utc"] == "2026-01-01T00:00:00+00:00"
    assert point["measurement_end_utc"] == "2026-01-01T00:00:10+00:00"


def test_cli_protocol_defaults_and_explicit_sweep() -> None:
    argv = ["--model", "qwen3-14b", "--variant", "awq_w4a16", "--tokenizer-path", "/local"]
    args = build_parser().parse_args(argv)
    assert args.input_tokens == 1000
    assert args.output_tokens == 256
    assert args.max_model_len == 8192
    assert args.concurrency_levels is None
    assert args.seed == 42
    assert args.dtype == "bfloat16"
    assert args.serving_seed == 42
    assert args.model_revision is None
    custom = build_parser().parse_args(argv + [
        "--dtype", "float16", "--seed", "17", "--serving-seed", "7",
        "--model-revision", "revision-from-cli",
    ])
    assert (custom.dtype, custom.seed, custom.serving_seed, custom.model_revision) == (
        "float16", 17, 7, "revision-from-cli",
    )
    assert build_parser().parse_args(argv + ["--model-revision", ""]).model_revision is None
    sweep = build_parser(sweep=True).parse_args(argv + ["--concurrency-levels", "1,4,8"])
    assert sweep.concurrency_levels == [1, 4, 8]


def test_runner_records_declared_serving_protocol(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = build_parser().parse_args([
        "--model", "model", "--variant", "bf16", "--tokenizer-path", "/local",
        "--num-requests", "1", "--warmup-requests", "0",
        "--dtype", "float16", "--seed", "17", "--serving-seed", "7",
        "--model-revision", "revision-from-cli",
        "--output-dir", str(tmp_path), "--run-id", "metadata",
    ])
    captured = {}

    async def fake_point(client, prompts, *, concurrency, warmup_requests, metadata):
        captured.update(metadata)
        return [], {
            **metadata, "model": client.model, "concurrency": concurrency,
            "slo_compliant_operating_point": False, "goodput_requests_per_second": 0,
        }

    class FakeHTTP:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    monkeypatch.setattr(runner, "load_local_tokenizer", lambda path: object())
    monkeypatch.setattr(runner, "build_workload", lambda *a, **kw: {"workload_sha256": "digest"})
    monkeypatch.setattr(runner, "validate_workload", lambda *a, **kw: prompts(1))
    monkeypatch.setattr(runner, "package_versions", lambda *names: {"torch": "test-version"})
    monkeypatch.setattr(runner, "save_json", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "run_point", fake_point)
    monkeypatch.setattr(runner.httpx, "AsyncClient", lambda **kw: FakeHTTP())
    asyncio.run(runner.run(args))
    assert captured["dtype_configuration"] == "float16"
    assert captured["serving_seed"] == 7
    assert captured["seed"] == 17
    assert captured["model_revision"] == "revision-from-cli"
    assert "immutable" in captured["model_revision_requirement"]
    assert captured["add_special_tokens"] is False


def test_unexpected_request_return_is_not_silently_dropped() -> None:
    async def invalid_send(*args):
        return None

    records = asyncio.run(closed_loop(prompts(3), 1, invalid_send, model="model"))
    assert len(records) == 3
    assert all(record.error_type == "TypeError" for record in records)
