"""Async scheduling with fake requests; no server or GPU."""

import asyncio
from collections import Counter

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
    records, point = asyncio.run(run_point(
        FakeClient(), prompts(8), concurrency=2, warmup_requests=3,
        metadata={}, clock=lambda: next(ticks),
    ))
    assert len(records) == point["num_requests"] == 5
    assert point["warmup_successful"] == 3
    assert point["benchmark_duration_seconds"] == 2.0
    assert point["throughput_requests_per_second"] == 2.5


def test_cli_protocol_defaults_and_explicit_sweep() -> None:
    argv = ["--model", "qwen3-14b", "--variant", "awq_w4a16", "--tokenizer-path", "/local"]
    args = build_parser().parse_args(argv)
    assert args.input_tokens == 1000
    assert args.output_tokens == 256
    assert args.max_model_len == 8192
    assert args.concurrency_levels is None
    assert args.seed == 42
    sweep = build_parser(sweep=True).parse_args(argv + ["--concurrency-levels", "1,4,8"])
    assert sweep.concurrency_levels == [1, 4, 8]


def test_unexpected_request_return_is_not_silently_dropped() -> None:
    async def invalid_send(*args):
        return None

    records = asyncio.run(closed_loop(prompts(3), 1, invalid_send, model="model"))
    assert len(records) == 3
    assert all(record.error_type == "TypeError" for record in records)
