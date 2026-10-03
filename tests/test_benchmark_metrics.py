"""Pure timing, rate, percentile and JSON checks."""

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from qwen_vllm_production.benchmark.metrics import RequestResult, aggregate, latency_metrics, percentile
from qwen_vllm_production.evaluation.common import save_json


def make_result(index: int, success: bool = True, compliant: bool = True) -> RequestResult:
    return RequestResult(
        index, "model", 4, index % 4, "2026-01-01T00:00:00+00:00", 1000,
        success=success, output_tokens=256 if success else None,
        ttft_ms=100.0 if compliant else 1200.0,
        tpot_ms_per_token=20.0, e2e_ms=5200.0,
        slo_compliant=success and compliant,
    )


def test_latency_definitions_and_single_token() -> None:
    assert latency_metrics(1.0, 1.2, 2.0, 5) == pytest.approx({
        "ttft_ms": 200.0, "e2e_ms": 1000.0, "tpot_ms_per_token": 200.0,
    })
    assert latency_metrics(1, 1.2, 2, 1)["tpot_ms_per_token"] is None
    assert latency_metrics(1, 1.2, 2, 0)["tpot_ms_per_token"] is None


def test_percentiles_are_linear_and_deterministic() -> None:
    assert percentile([40, 10, 30, 20], 0.5) == 25
    assert percentile([40, 10, 30, 20], 0.95) == pytest.approx(38.5)
    assert percentile([], 0.95) is None
    assert percentile([7], 0.95) == 7
    with pytest.raises(ValueError):
        percentile([float("nan")], 0.5)


def test_success_attainment_throughput_and_goodput() -> None:
    records = [make_result(0), make_result(1), make_result(2, compliant=False), make_result(3, success=False)]
    result = aggregate(records, model="model", concurrency=4, duration_seconds=2)
    assert result["request_success_rate"] == 0.75
    assert result["slo_attainment_rate"] == pytest.approx(2 / 3)
    assert result["throughput_requests_per_second"] == 1.5
    assert result["goodput_requests_per_second"] == 1.0
    assert result["output_tokens_per_second"] == 384.0
    assert result["num_failed"] == 1
    assert result["slo_compliant_operating_point"] is False


def test_failures_excluded_and_no_success_safe() -> None:
    success = make_result(0)
    failure = make_result(1, success=False)
    failure.ttft_ms = 100000
    point = aggregate([success, failure], model="model", concurrency=4, duration_seconds=1)
    assert point["p95_ttft_ms"] == 100
    empty = aggregate([failure], model="model", concurrency=4, duration_seconds=1)
    assert empty["slo_attainment_rate"] is None
    assert empty["p95_ttft_ms"] is None
    assert empty["request_success_rate"] == 0
    assert empty["goodput_requests_per_second"] == 0
    assert empty["slo_compliant_operating_point"] is False


def test_raw_and_aggregate_json(tmp_path: Path) -> None:
    record = make_result(0)
    record.requested_output_tokens = 256
    record.output_length_complete = True
    record.finish_reason = "length"
    record.stop_reason = None
    result = aggregate([record], model="model", concurrency=4, duration_seconds=1)
    save_json({"requests": [asdict(record)]}, tmp_path / "raw.json")
    save_json(result, tmp_path / "aggregate.json")
    raw = json.loads((tmp_path / "raw.json").read_text(encoding="utf-8"))
    saved = json.loads((tmp_path / "aggregate.json").read_text(encoding="utf-8"))
    assert raw["requests"][0]["output_tokens"] == 256
    assert raw["requests"][0]["requested_output_tokens"] == 256
    assert raw["requests"][0]["output_length_complete"] is True
    assert raw["requests"][0]["finish_reason"] == "length"
    assert raw["requests"][0]["stop_reason"] is None
    assert saved["slo_compliant_operating_point"] is True


def test_single_token_success_is_not_slo_compliant() -> None:
    record = make_result(0)
    record.output_tokens = 1
    record.tpot_ms_per_token = None
    record.slo_compliant = False
    point = aggregate([record], model="model", concurrency=4, duration_seconds=1)
    assert point["p95_tpot_ms_per_token"] is None
    assert point["slo_attainment_rate"] == 0
    assert point["slo_compliant_operating_point"] is False
