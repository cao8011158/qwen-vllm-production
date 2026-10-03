"""Client timing definitions and deterministic linear-interpolated percentiles."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable

from .slo import DEFAULT_SLO, PerformanceSLO, operating_point_compliant, request_compliant


@dataclass
class RequestResult:
    request_id: int
    model: str
    concurrency: int
    worker_id: int
    start_timestamp: str
    prompt_tokens: int
    success: bool = False
    http_status: int | None = None
    output_tokens: int | None = None
    output_token_count_source: str | None = None
    prompt_token_count_source: str = "local_tokenizer"
    ttft_ms: float | None = None
    tpot_ms_per_token: float | None = None
    e2e_ms: float | None = None
    slo_compliant: bool = False
    error_type: str | None = None
    error_message: str | None = None
    first_content_timestamp: str | None = None
    end_timestamp: str | None = None
    prompt_sha256: str | None = None
    generated_text: str = ""
    request_start_monotonic_seconds: float | None = None
    first_content_monotonic_seconds: float | None = None
    stream_complete_monotonic_seconds: float | None = None


def latency_metrics(start: float, first: float, end: float, output_tokens: int) -> dict:
    if not all(math.isfinite(value) for value in (start, first, end)) or not start <= first <= end:
        raise ValueError("Timing must be finite and ordered: start <= first <= end.")
    if output_tokens < 0:
        raise ValueError("output_tokens must be non-negative.")
    ttft = (first - start) * 1000.0
    e2e = (end - start) * 1000.0
    return {
        "ttft_ms": ttft,
        "e2e_ms": e2e,
        "tpot_ms_per_token": (e2e - ttft) / (output_tokens - 1) if output_tokens > 1 else None,
    }


def percentile(values: Iterable[float], quantile: float) -> float | None:
    """Type 7: linearly interpolate sorted values at rank (n - 1) * q."""
    if not 0 <= quantile <= 1:
        raise ValueError("quantile must be in [0, 1].")
    ordered = sorted(values)
    if not ordered:
        return None
    if any(not math.isfinite(value) or value < 0 for value in ordered):
        raise ValueError("Latencies must be finite and non-negative.")
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def aggregate(
    requests: list[RequestResult], *, model: str, concurrency: int,
    duration_seconds: float, metadata: dict | None = None,
    slo: PerformanceSLO = DEFAULT_SLO,
) -> dict:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("Measured wall-clock duration must be positive and finite.")
    if any(r.model != model or r.concurrency != concurrency for r in requests):
        raise ValueError("Request records must belong to the same operating point.")
    successful = [r for r in requests if r.success]
    compliant = sum(request_compliant(r.success, r.ttft_ms, r.tpot_ms_per_token, r.e2e_ms, slo)
                    for r in successful)
    result = {
        **(metadata or {}),
        "model": model, "concurrency": concurrency,
        "num_requests": len(requests), "num_successful": len(successful),
        "num_failed": len(requests) - len(successful),
        "benchmark_duration_seconds": duration_seconds,
        "percentile_method": "linear interpolation at (n - 1) * q (type 7)",
        "request_success_rate": len(successful) / len(requests) if requests else 0.0,
        "slo_attainment_rate": compliant / len(successful) if successful else None,
        "num_slo_compliant": compliant,
        "throughput_requests_per_second": len(successful) / duration_seconds,
        "output_tokens_per_second": (
            sum(r.output_tokens for r in successful) / duration_seconds
            if all(r.output_tokens is not None for r in successful) else None
        ),
        "goodput_requests_per_second": compliant / duration_seconds,
        "slo_thresholds": asdict(slo),
    }
    for field in ("ttft_ms", "tpot_ms_per_token", "e2e_ms"):
        values = [getattr(r, field) for r in successful if getattr(r, field) is not None]
        result["p50_" + field] = percentile(values, 0.50)
        result["p95_" + field] = percentile(values, 0.95)
    result["slo_compliant_operating_point"] = operating_point_compliant(result, slo)
    return result
