"""Per-request and operating-point SLO boundaries."""

import pytest

from qwen_vllm_production.benchmark.slo import operating_point_compliant, request_compliant


def test_request_inclusive_boundaries_and_missing_tpot() -> None:
    assert request_compliant(True, 1000, 50, 15000)
    assert not request_compliant(True, 1000, None, 15000)
    assert not request_compliant(False, 100, 10, 1000)


@pytest.mark.parametrize("key,value", [
    ("p95_ttft_ms", 1000.1), ("p95_tpot_ms_per_token", 50.1),
    ("p95_e2e_ms", 15000.1), ("request_success_rate", 0.989),
    ("slo_attainment_rate", 0.949), ("p95_tpot_ms_per_token", None),
])
def test_operating_point_failures(key: str, value: float | None) -> None:
    point = {
        "p95_ttft_ms": 1000, "p95_tpot_ms_per_token": 50,
        "p95_e2e_ms": 15000, "request_success_rate": 0.99, "slo_attainment_rate": 0.95,
    }
    assert operating_point_compliant(point)
    point[key] = value
    assert not operating_point_compliant(point)
