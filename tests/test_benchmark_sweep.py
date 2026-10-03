"""Independent maxima; no fabricated capacity when all points fail."""

import pytest

from qwen_vllm_production.benchmark.sweep import summarize_points


def point(concurrency: int, goodput: float, compliant: bool = True) -> dict:
    return {
        "model": "model", "concurrency": concurrency,
        "slo_compliant_operating_point": compliant,
        "goodput_requests_per_second": goodput,
    }


def test_capacity_and_goodput_can_peak_at_different_concurrency() -> None:
    summary = summarize_points([point(1, 2), point(4, 8), point(8, 6), point(16, 10, False)])
    assert summary["maximum_slo_compliant_concurrency"] == 8
    assert summary["maximum_slo_compliant_goodput"] == 8
    assert summary["concurrency_at_maximum_slo_compliant_goodput"] == 4


def test_no_compliant_operating_point_and_tie_break() -> None:
    summary = summarize_points([point(1, 0, False), point(4, 0, False)])
    assert summary["maximum_slo_compliant_concurrency"] is None
    assert summary["maximum_slo_compliant_goodput"] is None
    assert summary["concurrency_at_maximum_slo_compliant_goodput"] is None
    tied = summarize_points([point(4, 8), point(8, 8)])
    assert tied["concurrency_at_maximum_slo_compliant_goodput"] == 4


def test_mixed_protocols_are_rejected() -> None:
    with pytest.raises(ValueError, match="variant"):
        summarize_points([{**point(1, 2), "variant": "bf16"}, {**point(4, 8), "variant": "awq_w4a16"}])
