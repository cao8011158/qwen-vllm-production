"""Per-request and operating-point SLO boundaries."""

from pathlib import Path

import pytest
import yaml

from qwen_vllm_production.benchmark.slo import operating_point_compliant, request_compliant


def test_request_inclusive_boundaries_and_missing_tpot() -> None:
    assert request_compliant(True, 1000, 50, 15000)
    assert not request_compliant(True, 1000, None, 15000)
    assert not request_compliant(False, 100, 10, 1000)


@pytest.mark.parametrize("metric_index", [0, 1, 2])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -float("inf"), -0.1])
def test_request_rejects_missing_nonfinite_and_negative_latencies(
    metric_index: int, value: float | None,
) -> None:
    latencies = [1000.0, 50.0, 15000.0]
    latencies[metric_index] = value
    assert not request_compliant(True, *latencies)


def test_operating_point_inclusive_boundaries_without_percentiles() -> None:
    assert operating_point_compliant({
        "request_success_rate": 0.99, "slo_attainment_rate": 0.95,
    })


@pytest.mark.parametrize("key,value", [
    ("request_success_rate", 0.989), ("slo_attainment_rate", 0.949),
])
def test_operating_point_rates_below_threshold(key: str, value: float) -> None:
    point = {"request_success_rate": 0.99, "slo_attainment_rate": 0.95}
    point[key] = value
    assert not operating_point_compliant(point)


@pytest.mark.parametrize("key", ["request_success_rate", "slo_attainment_rate"])
def test_operating_point_missing_rate(key: str) -> None:
    point = {"request_success_rate": 0.99, "slo_attainment_rate": 0.95}
    del point[key]
    assert not operating_point_compliant(point)


@pytest.mark.parametrize("key", ["request_success_rate", "slo_attainment_rate"])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -float("inf")])
def test_operating_point_invalid_rate(key: str, value: float | None) -> None:
    point = {"request_success_rate": 0.99, "slo_attainment_rate": 0.95}
    point[key] = value
    assert not operating_point_compliant(point)


def test_operating_point_percentiles_are_descriptive_only() -> None:
    point = {
        "request_success_rate": 0.99, "slo_attainment_rate": 0.95,
        "p95_ttft_ms": 1000.1, "p95_tpot_ms_per_token": 50.1, "p95_e2e_ms": 15000.1,
    }
    assert operating_point_compliant(point)


def test_project_latency_config_uses_request_level_thresholds() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["slo"]["latency"] == {
        "ttft_seconds": 1.0, "tpot_ms_per_token": 50, "e2e_seconds": 15,
    }
