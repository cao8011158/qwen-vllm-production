"""Mock Grafana provisioning and its Prometheus proxy query API."""

import asyncio

import httpx
import pytest

from qwen_vllm_production.observability.checks import check_grafana


@pytest.mark.parametrize("scenario,expected", [
    ("healthy", True), ("unhealthy", False), ("datasource_missing", False),
    ("datasource_wrong_type", False), ("dashboard_missing", False),
    ("query_failure", False), ("query_empty", False),
])
def test_grafana_validation(scenario: str, expected: bool) -> None:
    async def invoke():
        def handle(request):
            path = request.url.path
            if path == "/api/health":
                return httpx.Response(200, json={"database": "failed" if scenario == "unhealthy" else "ok"})
            if path == "/api/datasources/uid/vllm-prometheus":
                if scenario == "datasource_missing":
                    return httpx.Response(404)
                return httpx.Response(200, json={
                    "uid": "vllm-prometheus",
                    "type": "other" if scenario == "datasource_wrong_type" else "prometheus",
                })
            if path == "/api/dashboards/uid/vllm-serving-overview":
                if scenario == "dashboard_missing":
                    return httpx.Response(404)
                return httpx.Response(200, json={"dashboard": {
                    "uid": "vllm-serving-overview",
                    "panels": [{"targets": [{"expr": "vllm:num_requests_running"}]}],
                }})
            if path == "/api/datasources/proxy/uid/vllm-prometheus/api/v1/query":
                if scenario == "query_failure":
                    return httpx.Response(502)
                return httpx.Response(200, json={"status": "success", "data": {
                    "resultType": "vector",
                    "result": [] if scenario == "query_empty" else [{"value": [1, "0"]}],
                }})
            return httpx.Response(404)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await check_grafana(client, "http://grafana")

    assert asyncio.run(invoke()).ok is expected
