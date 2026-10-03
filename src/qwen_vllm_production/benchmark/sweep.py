"""Closed-loop scheduling and separate capacity and goodput maxima."""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

from .client import utc_now
from .metrics import RequestResult
from .workload import Prompt


async def closed_loop(
    prompts: list[Prompt], concurrency: int,
    send: Callable[[Prompt, int, int, int], Awaitable[RequestResult]],
    *, model: str,
) -> list[RequestResult]:
    if concurrency <= 0:
        raise ValueError("concurrency must be positive.")
    next_index = 0
    results: list[RequestResult | None] = [None] * len(prompts)

    async def worker(worker_id: int) -> None:
        nonlocal next_index
        while next_index < len(prompts):
            # No await between claiming and incrementing: deterministic request order.
            index = next_index
            next_index += 1
            prompt = prompts[index]
            started_at = utc_now()
            started = time.perf_counter()
            try:
                result = await send(prompt, index, worker_id, concurrency)
                if not isinstance(result, RequestResult):
                    raise TypeError("Request function must return a RequestResult.")
            except Exception as exc:
                result = RequestResult(
                    index, model, concurrency, worker_id, started_at, prompt.prompt_tokens,
                    error_type=type(exc).__name__, error_message=str(exc) or type(exc).__name__,
                    prompt_sha256=prompt.sha256,
                    e2e_ms=(time.perf_counter() - started) * 1000.0,
                    end_timestamp=utc_now(),
                )
            results[index] = result

    await asyncio.gather(*(worker(index) for index in range(concurrency)))
    return [result for result in results if result is not None]


def summarize_points(points: list[dict]) -> dict:
    if not points:
        raise ValueError("At least one operating point is required.")
    controls = (
        "model", "variant", "seed", "target_input_tokens", "target_output_tokens",
        "max_model_len", "workload_sha256", "slo_thresholds", "monitoring_enabled",
        "prometheus_scrape_interval_seconds", "ignore_eos", "request_path",
        "num_requests", "warmup_requests",
        "tensor_parallel_size", "gpu_memory_utilization", "dtype_configuration",
        "checkpoint_path", "timeout_seconds",
        "prefix_caching",
        "base_url", "package_versions", "tokenizer_path", "temperature",
        "chat_template_applied_locally", "enable_thinking", "serving_seed",
        "model_revision", "add_special_tokens",
    )
    for key in controls:
        if any((key in point) != (key in points[0]) or point.get(key) != points[0].get(key)
               for point in points):
            raise ValueError(f"Cannot summarize different experimental protocols: {key}.")
    if len({point["concurrency"] for point in points}) != len(points):
        raise ValueError("Each concurrency must appear once in a sweep summary.")
    ordered = sorted(points, key=lambda point: point["concurrency"])
    compliant = [point for point in ordered if point["slo_compliant_operating_point"]]
    best = max(compliant, key=lambda point: (point["goodput_requests_per_second"], -point["concurrency"])) if compliant else None
    return {
        "operating_points": ordered,
        "maximum_slo_compliant_concurrency": max(point["concurrency"] for point in compliant) if compliant else None,
        "maximum_slo_compliant_goodput": best["goodput_requests_per_second"] if best else None,
        "concurrency_at_maximum_slo_compliant_goodput": best["concurrency"] if best else None,
        "goodput_tie_break": "lower concurrency",
    }
