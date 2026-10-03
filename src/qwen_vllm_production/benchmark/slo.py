"""Performance SLOs; Phase 2 quality is not recalculated here."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PerformanceSLO:
    ttft_ms: float = 1000.0
    tpot_ms_per_token: float = 50.0
    e2e_ms: float = 15000.0
    request_success_rate: float = 0.99
    attainment_rate: float = 0.95


DEFAULT_SLO = PerformanceSLO()


def request_compliant(
    success: bool, ttft_ms: float | None, tpot_ms_per_token: float | None,
    e2e_ms: float | None, slo: PerformanceSLO = DEFAULT_SLO,
) -> bool:
    values = (ttft_ms, tpot_ms_per_token, e2e_ms)
    if not success or any(value is None or not math.isfinite(value) or value < 0 for value in values):
        return False
    return ttft_ms <= slo.ttft_ms and tpot_ms_per_token <= slo.tpot_ms_per_token and e2e_ms <= slo.e2e_ms


def operating_point_compliant(result: dict, slo: PerformanceSLO = DEFAULT_SLO) -> bool:
    checks = (
        ("p95_ttft_ms", slo.ttft_ms, False),
        ("p95_tpot_ms_per_token", slo.tpot_ms_per_token, False),
        ("p95_e2e_ms", slo.e2e_ms, False),
        ("request_success_rate", slo.request_success_rate, True),
        ("slo_attainment_rate", slo.attainment_rate, True),
    )
    for key, threshold, minimum in checks:
        value = result.get(key)
        if value is None or not math.isfinite(value):
            return False
        if (minimum and value < threshold) or (not minimum and value > threshold):
            return False
    return True
