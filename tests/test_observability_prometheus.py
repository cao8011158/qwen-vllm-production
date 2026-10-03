"""Mock Prometheus health/ready/targets/query, including failures."""

import asyncio

import httpx
import pytest

from qwen_vllm_production.observability.checks import check_prometheus


def vector(values: list[str]) -> dict:
    return {"status": "success", "data": {
        "resultType": "vector",
        "result": [{"metric": {}, "value": [1, value]} for value in values],
    }}


@pytest.mark.parametrize("scenario,expected", [
    ("up", True), ("down", False), ("missing", False), ("empty", False),
    ("api_error", False), ("timeout", False), ("query_failure", False),
    ("not_ready", False), ("nan", False),
])
def test_prometheus_validation(scenario: str, expected: bool) -> None:
    async def invoke():
        def handle(request):
            path = request.url.path
            if scenario == "timeout" and path == "/-/healthy":
                raise httpx.ReadTimeout("fake timeout")
            if scenario == "not_ready" and path == "/-/ready":
                return httpx.Response(503)
            if path == "/api/v1/targets":
                targets = [] if scenario == "missing" else [{
                    "labels": {"job": "vllm"},
                    "health": "down" if scenario == "down" else "up",
                    "lastError": "",
                }]
                return httpx.Response(200, json={"status": "success", "data": {"activeTargets": targets}})
            if path == "/api/v1/query":
                assert request.url.params["query"] == "vllm:num_requests_running"
                if scenario == "api_error":
                    return httpx.Response(500)
                if scenario == "query_failure":
                    return httpx.Response(200, json={"status": "error", "error": "bad query"})
                return httpx.Response(200, json=vector(
                    [] if scenario == "empty" else ["NaN" if scenario == "nan" else "1"]
                ))
            return httpx.Response(200, text="OK")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await check_prometheus(client, "http://prometheus")

    report = asyncio.run(invoke())
    assert report.ok is expected
    assert len(report.checks) == 4
