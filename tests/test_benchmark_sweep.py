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


@pytest.mark.parametrize("key,first,second", [
    ("base_url", "http://first", "http://second"),
    ("package_versions", {"torch": "first"}, {"torch": "second"}),
    ("tokenizer_path", "/first", "/second"),
    ("temperature", 0.0, 1.0),
    ("chat_template_applied_locally", True, False),
    ("enable_thinking", False, True),
    ("dtype_configuration", "bfloat16", "float16"),
    ("serving_seed", 42, 7),
    ("model_revision", "first-revision", "second-revision"),
    ("add_special_tokens", False, True),
])
def test_serving_protocol_mismatch_is_rejected(key, first, second) -> None:
    with pytest.raises(ValueError, match=key):
        summarize_points([{**point(1, 2), key: first}, {**point(4, 8), key: second}])
    with pytest.raises(ValueError, match=key):
        summarize_points([{**point(1, 2), key: first}, point(4, 8)])
    summary = summarize_points([{**point(1, 2), key: first}, {**point(4, 8), key: first}])
    assert summary["maximum_slo_compliant_concurrency"] == 4
    assert summary["maximum_slo_compliant_goodput"] == 8


def test_explicit_empty_revision_cannot_mix_with_missing_metadata() -> None:
    with pytest.raises(ValueError, match="model_revision"):
        summarize_points([{**point(1, 2), "model_revision": None}, point(4, 8)])
