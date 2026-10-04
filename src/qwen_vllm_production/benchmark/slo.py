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
    """Gate on reliability and joint request-level attainment, not percentiles."""
    success_rate = result.get("request_success_rate")
    attainment_rate = result.get("slo_attainment_rate")
    if any(value is None or not math.isfinite(value) for value in (success_rate, attainment_rate)):
        return False
    return success_rate >= slo.request_success_rate and attainment_rate >= slo.attainment_rate
