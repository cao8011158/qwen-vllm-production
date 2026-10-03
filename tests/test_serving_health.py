"""Mock health/models/metrics checks."""

import asyncio

import httpx
import pytest

from qwen_vllm_production.observability.checks import check_serving


@pytest.mark.parametrize("models,metrics,healthy", [
    (["model"], "# TYPE vllm:num_requests_running gauge\nvllm:num_requests_running 0\n", True),
    (["other"], "# TYPE vllm:num_requests_running gauge\n", False),
    (["model"], "not metrics", False),
])
def test_serving_endpoints(models: list[str], metrics: str, healthy: bool) -> None:
    async def invoke():
        def handle(request):
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": name} for name in models]})
            return httpx.Response(200, text=metrics if request.url.path == "/metrics" else "")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await check_serving(client, "http://fake", "model")

    assert asyncio.run(invoke()).ok is healthy
